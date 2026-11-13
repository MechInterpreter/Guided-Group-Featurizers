import copy, gc, hashlib, json, math, os, time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training, metrics
from coco_baselines.storage import identity, sha_file
from coco_adaptation import core
from coco_coordinated import run as memory
from coco_ggf import study
from coco_ggf import matched_baselines as matched
from scale_metrics import checked
from coco_ggf.slics_repair import Solver
from coco_ggf.slics_closeout import support, atom_sweep
from scale_data import mask_dictionary
from scale_models import model
from supervised_fit import fit
from precision_retry import PrecisionRetry
from training_precision import (
    JointPrecisionRetry,
    precision_atom_sweep as _precision_atom,
)

G = 19
K = 8
SEEDS = [21, 22, 23]
WARM = 7640
GUIDED = 3000
TOTAL = WARM + GUIDED
SOLVER = dict(nnls_iterations=2000, nnls_tolerance=1e-5, nnls_fallback_iterations=20000)


def native_solver_config():
    return dict(
        nnls_iterations=2000, nnls_tolerance=1e-5, nnls_fallback_iterations=20000
    )


def row_from_state(ctx, key, seed, arm, groups, targets, st, **extra):
    r = dict(
        key=key,
        seed=seed,
        arm=arm,
        method="released_vanilla",
        groups=groups,
        rank=12,
        active_groups=K,
        weights_sha=training.weight_sha(st["weights"]),
        targets=targets,
        targets_sha=identity(targets),
        total_training_updates=TOTAL,
        selected_total_updates=extra.pop("selected_total_updates", TOTAL),
        **extra,
    )
    r["development"] = evaluate_state(ctx, r, ctx["packs"]["select"], False)
    return r


def native_initializer(ctx):
    key = "native_svd12"
    old = ctx["store"].load(key)
    if old is not None:
        return old["D"]
    D = mask_dictionary(ctx).clone().cpu()
    b = 12
    if D.shape != (G * 12, ctx["packs"]["train"]["x"].shape[-1]):
        raise ValueError("Expected Gx12 native initializer")
    # Same uncentered SVD vectors; choose native SLiCS polarity by positive coefficient energy.
    for j, c in enumerate(ctx["cfg"]["category_indices"]):
        pos = np.zeros(b)
        neg = np.zeros(b)
        block = D[j * b : (j + 1) * b].numpy()
        for x, mm in zip(ctx["packs"]["train"]["x"], ctx["packs"]["train"]["masks"]):
            good = (mm["crowd"] <= 0.1) & (
                mm["fraction"][:, c].astype(float) / 255 >= 0.5
            )
            values = np.asarray(x[good], np.float64) @ block.astype(np.float64).T
            pos += np.square(np.maximum(values, 0)).sum(0)
            neg += np.square(np.minimum(values, 0)).sum(0)
        D[j * b : (j + 1) * b] *= torch.as_tensor(
            np.where(pos >= neg, 1.0, -1.0), dtype=D.dtype
        )[:, None]
    ctx["store"].save(key, dict(D=D, sha=training.weight_sha({"D": D})))
    return D


def small_context(ctx):
    rule = dict(
        ctx["core_rule"],
        snr=0.0,
        warm_updates=WARM,
        final_guided_updates=GUIDED,
        sampling="epochs",
        known_only=True,
    )
    return dict(ctx, rule=rule)


def small_warm(ctx, seed):
    sub = small_context(ctx)
    key = f"smallG_s{seed}_warm"
    torch.manual_seed(seed)
    m = model("released_vanilla", G, 12, K, ctx["packs"]["train"]["x"].shape[-1])
    D = native_initializer(ctx)
    with torch.no_grad():
        m.W_dec.copy_(D)
        m.W_enc.copy_(D.T)
        m.b_enc.zero_()
    start = training.weights(m)
    del m
    st = fit(sub, key, "released_vanilla", G, K, seed, start, WARM, WARM, stream=0)
    targetkey = f"smallG_s{seed}_targets"
    targets = ctx["store"].load(targetkey)
    if targets is None:
        targets = fixed_targets(ctx)
        m = model("released_vanilla", G, 12, K, D.shape[1]).to(ctx["device"])
        m.load_state_dict(st["weights"])
        count = 0
        total = np.zeros(G)
        square = np.zeros(G)
        with torch.no_grad():
            for x in ctx["packs"]["train"]["x"]:
                h = (
                    m.encode(
                        torch.as_tensor(np.array(x, copy=True), device=ctx["device"])
                    )
                    .norm(dim=-1)
                    .cpu()
                    .numpy()
                    .astype(float)
                )
                count += len(h)
                total += h.sum(0)
                square += np.square(h).sum(0)
        mean = total / count
        sd = np.sqrt(np.maximum(square / count - mean**2, 0))
        for j, t in enumerate(targets):
            t.update(heat_mean=float(mean[j]), heat_scale=max(float(sd[j]), 1e-3))
        ctx["store"].save(targetkey, targets)
        del m
        gc.collect()
    return st, targets


def small_fit(ctx, seed, lr, policy="object"):
    key = f"smallG_s{seed}_lr{int(lr * 1e7)}_ggf"
    cache = ctx["store"].load("row_" + key + "_" + policy)
    if cache is not None:
        return cache
    if policy != "object":
        raise ValueError("Only the object selector is supported")
    sub = small_context(ctx)
    warm, targets = small_warm(ctx, seed)
    st = fit(
        sub,
        key,
        "released_vanilla",
        G,
        K,
        seed,
        warm["weights"],
        GUIDED,
        GUIDED,
        targets=targets,
        cap=0.6,
        learning_rate=lr,
        stream=1,
        keep_candidates=True,
        readout_targets=targets,
    )
    chosen = memory.required(ctx["store"], key + "_" + policy)
    r = row_from_state(
        ctx,
        key + "_" + policy,
        seed,
        "matched_ggf",
        G,
        targets,
        chosen,
        selected_total_updates=WARM + chosen["updates"],
        lr=lr,
        head_parameters=2 * G,
        fitted_key=key,
        selection_policy=policy,
    )
    ctx["store"].save("row_" + key + "_" + policy, r)
    del st, warm
    gc.collect()
    return r


@torch.no_grad()
def partial_nnls(x, D, cfg, width):
    """Vectorized Eq.3: independent small NNLS per concept, never sum as a reconstruction."""
    groups = len(D) // width
    B = D.reshape(groups, width, -1)
    gram = B @ B.transpose(1, 2)
    cross = (x @ D.T).reshape(len(x), groups, width)
    L = torch.linalg.eigvalsh(gram).amax(-1).clamp_min(1e-8)[None, :, None]
    z = torch.zeros_like(cross)
    y = z.clone()
    t = 1.0
    for it in range(cfg["nnls_iterations"]):
        grad = torch.einsum("ngb,gbc->ngc", y, gram) - cross
        new = (y - grad / L).clamp_min(0)
        nt = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = new + (t - 1) / nt * (new - z)
        z = new
        t = nt
        if (it + 1) % 50 == 0:
            grad = torch.einsum("ngb,gbc->ngc", z, gram) - cross
            error = (z - (z - grad).clamp_min(0)).abs().amax(-1) / cross.abs().amax(
                -1
            ).clamp_min(1e-12)
            if float(error.max()) <= cfg["nnls_tolerance"] * 0.5:
                break
    BB = B.double()
    gram64 = BB @ BB.transpose(1, 2)
    cross64 = (x.double() @ D.double().T).reshape_as(cross)

    def errors():
        zz = z.double()
        grad = torch.einsum("ngb,gbc->ngc", zz, gram64) - cross64
        return (zz - (zz - grad).clamp_min(0)).abs().amax(-1) / cross64.abs().amax(
            -1
        ).clamp_min(1e-12)

    bad = torch.nonzero(errors() > cfg["nnls_tolerance"])
    from scipy.optimize import nnls

    for i, g in bad.tolist():
        result, _ = nnls(
            B[g].double().cpu().numpy().T,
            x[i].double().cpu().numpy(),
            maxiter=cfg["nnls_fallback_iterations"],
        )
        z[i, g] = torch.as_tensor(result, device=z.device, dtype=z.dtype)
    error = errors()
    if not torch.isfinite(z).all() or float(error.max()) > cfg["nnls_tolerance"]:
        raise RuntimeError("Partial NNLS failed strict KKT verification")
    return z, dict(cpu_fallback_rows=len(bad), max_relative_kkt=float(error.max()))


class NativeSLiCS(torch.nn.Module):
    def __init__(self, D, cfg, readout="joint"):
        super().__init__()
        self.register_buffer("D", D)
        self.d = D.shape[1]
        self.group_size = 12
        self.n_groups = len(D) // 12
        self.cfg = cfg
        self.readout = readout
        self.solver = Solver(D, cfg)
        self.stats = dict(
            cpu_fallback_rows=0,
            calls=0,
            max_relative_kkt=0.0,
            readout_rows=0,
            readout_active_group_sum=0.0,
        )

    def atoms(self):
        return self.D.reshape(self.n_groups, 12, self.d)

    def solve(self, x, D=None):
        solver = self.solver if D is None else Solver(D, self.cfg)
        z, info = solver.solve(x)
        self.stats["calls"] += 1
        self.stats["cpu_fallback_rows"] += info["cpu_fallback_rows"]
        self.stats["max_relative_kkt"] = max(
            self.stats["max_relative_kkt"], info["max_relative_kkt"]
        )
        return z

    @torch.no_grad()
    def encode(self, x):
        if self.readout == "joint":
            z = self.solve(x).reshape(-1, self.n_groups, 12)
        elif self.readout == "partial":
            z, info = partial_nnls(x, self.D, self.cfg, 12)
            self.stats["calls"] += self.n_groups
            self.stats["cpu_fallback_rows"] += info["cpu_fallback_rows"]
            self.stats["max_relative_kkt"] = max(
                self.stats["max_relative_kkt"], info["max_relative_kkt"]
            )
        else:
            raise ValueError("Unknown SLiCS readout")
        self.stats["readout_rows"] += len(x)
        self.stats["readout_active_group_sum"] += float((z.norm(dim=-1) > 1e-6).sum())
        return z

    @torch.no_grad()
    def forward(self, x):
        # Eq.3 partial concept projections are NOT additive reconstructions.
        z = self.solve(x).reshape(-1, self.n_groups, 12)
        return z.flatten(1) @ self.D.to(z.dtype), z


def batch_indices(n, seed, u):
    stream = 0 if u < WARM else 1
    offset = u if stream == 0 else u - WARM
    from supervised_fit import epoch_indices

    return epoch_indices(n, seed, offset, stream)


def native_fit(ctx, seed):
    s = ctx["store"]
    key = f"native_slics12_s{seed}"
    prior_rows = s.load("native_rows_s" + str(seed))
    if prior_rows is not None:
        return prior_rows
    cfg = native_solver_config()
    D0 = native_initializer(ctx)
    ident = identity(
        dict(
            source=training.weight_sha({"D": D0}),
            train=ctx["packs"]["train"]["identity"],
            select=ctx["packs"]["select"]["identity"],
            seed=seed,
            solver=cfg,
            total=TOTAL,
            protocol=protocol(),
        )
    )
    st = s.load(key)
    if st is not None and st["fingerprint"] != ident:
        raise ValueError("SLiCS resume identity changed")
    if st is None:
        st = dict(
            fingerprint=ident,
            updates=0,
            target=TOTAL,
            D=D0,
            history=[],
            seconds=0.0,
            selected={},
            sampled_batches_sha=hashlib.sha256().hexdigest(),
            phase_batch_shas={},
        )

    def save():
        s._width_drive_recovery_original_save(key, st, drive=True)

    def score():
        u = st["updates"]
        D = st["D"].to(ctx["device"])
        m = NativeSLiCS(D, cfg, "partial")
        val = checked_native_evaluate(ctx, m, ctx["packs"]["select"], False)
        for policy, value in [("object", val["macro_capture"] + 0.2 * val["macro_ap"])]:
            if policy not in st["selected"] or value > st["selected"][policy]["score"]:
                st["selected"][policy] = dict(score=value, updates=u)
                s.save(
                    key + "_" + policy,
                    dict(D=st["D"].clone(), updates=u, validation=val, score=value),
                )
        st["history"].append(
            dict(update=u, validation=val, solver=copy.deepcopy(m.stats))
        )
        st["last_evaluated"] = u
        s.write("histories/" + key + ".json", st["history"])
        save()
        del m, D
        gc.collect()

    # O(offset) schedule replay is avoided in the training loop by restoring NumPy state.
    rng = np.random.default_rng(126000 + seed + 100000 * (st["updates"] >= WARM))
    if "sample_rng" in st:
        rng.bit_generator.state = st["sample_rng"]
    elif st["updates"]:
        raise ValueError("Missing exact sampling RNG")
    try:
        if st.get("last_evaluated") != st["updates"] and (
            st["updates"] % 500 == 0 or st["updates"] in [WARM, TOTAL]
        ):
            score()
        while st["updates"] < TOTAL:
            u = st["updates"]
            if u == WARM and not st.get("phase_reset"):
                rng = np.random.default_rng(126000 + seed + 100000)
                st["phase_reset"] = True
                st["phase_batch_shas"]["warm"] = st["sampled_batches_sha"]
                st["sampled_batches_sha"] = hashlib.sha256().hexdigest()
                st["sample_rng"] = copy.deepcopy(rng.bit_generator.state)
            rng_before = copy.deepcopy(rng.bit_generator.state)
            ids = batch_indices(len(ctx["packs"]["train"]["ids"]), seed, u)
            pack = ctx["packs"]["train"]
            xx = torch.as_tensor(
                np.array(pack["x"][ids], copy=True), device=ctx["device"]
            ).reshape(-1, D0.shape[1])
            allow, known = support(
                [pack["masks"][i] for i in ids],
                ctx["cfg"]["category_indices"],
                dict(rank=12),
                "mask",
                ctx["device"],
            )
            take = torch.as_tensor(known, device=ctx["device"])
            xx = xx[take]
            allow = allow[take]
            D = st["D"].to(ctx["device"]).clone()
            start = time.monotonic()
            try:
                # Native training support: remove globally forbidden columns before any Gram allocation.
                cols = allow.any(0)
                z = torch.zeros((len(xx), G * 12), device=ctx["device"])
                info = dict(
                    cpu_fallback_rows=0, max_relative_kkt=0.0, unsupported_rows=len(xx)
                )
                if bool(cols.any()):
                    zz, info = Solver(D[cols], cfg).solve(xx, allow[:, cols])
                    z = z.to(zz.dtype)
                    z[:, cols] = zz
                changed, atom = _precision_atom(atom_sweep, xx, D, z, allow, False)
                if not torch.isfinite(changed).all():
                    raise FloatingPointError("SLiCS atoms")
                torch.cuda.synchronize() if str(ctx["device"]).startswith(
                    "cuda"
                ) else None
            except BaseException:
                rng.bit_generator.state = rng_before
                raise
            st.update(
                D=changed.cpu(),
                updates=u + 1,
                seconds=st["seconds"] + time.monotonic() - start,
                sample_rng=copy.deepcopy(rng.bit_generator.state),
                sampled_batches_sha=hashlib.sha256(
                    st["sampled_batches_sha"].encode() + ids.tobytes()
                ).hexdigest(),
            )
            st.setdefault("optimization_history", []).append(
                dict(update=u + 1, solver=info, atom=atom)
            )
            if st["updates"] % 500 == 0 or st["updates"] in [WARM, TOTAL]:
                score()
            elif st["updates"] % 50 == 0:
                s.save(key, st)
                s.event(
                    "slics_progress",
                    seed=seed,
                    update=st["updates"],
                    total=TOTAL,
                    step_seconds=st["seconds"] / st["updates"],
                    solver=info,
                )
            del D, z, xx, allow
            gc.collect() if st["updates"] % 250 == 0 else None
        save()
    except BaseException:
        save()
        raise
    result = {}
    for policy in ["object"]:
        chosen = memory.required(s, key + "_" + policy)
        for readout in ["partial"]:
            name = f"slics_{policy}_{readout}"
            r = dict(
                key=key + "_" + policy,
                seed=seed,
                arm=name,
                method="native_slics",
                groups=G,
                rank=12,
                active_groups=None,
                weights_sha=training.weight_sha({"D": chosen["D"]}),
                targets=fixed_targets(ctx),
                targets_sha=identity(fixed_targets(ctx)),
                total_training_updates=TOTAL,
                selected_total_updates=chosen["updates"],
                readout=readout,
                solver=cfg,
                head_parameters=0,
                training_seconds=st["seconds"],
                selection_policy=policy,
            )
            r["development"] = evaluate_state(ctx, r, ctx["packs"]["select"], False)
            result[name] = r
    s.save("native_rows_s" + str(seed), result)
    return result


def checked_native_evaluate(ctx, m, pack, compact):
    # Exact existing object metrics; joint reconstruction is explicit for both native readouts.
    return evaluate_model(
        m,
        pack,
        fixed_targets(ctx),
        ctx["cfg"],
        ctx["device"],
        compact=compact,
        reconstruction=False,
    )


def pack_variance(pack):
    if "_global_variance" in pack:
        return pack["_global_variance"]
    a = pack["x"].reshape(-1, pack["x"].shape[-1])
    total = np.zeros(a.shape[1])
    energy = 0.0
    for lo in range(0, len(a), 512):
        chunk = np.asarray(a[lo : lo + 512], np.float64)
        total += chunk.sum(0)
        energy += float(np.square(chunk).sum())
    result = max(energy - float(total @ total) / len(a), 1e-12)
    pack["_global_variance"] = result
    return result


@torch.no_grad()
def evaluate_model(m, pack, targets, cfg, device, compact=False, reconstruction=True):
    """Existing metric arithmetic, one bounded patch pass and streaming global variance."""
    from scale_metrics import ReadoutView, checked

    class Meter:
        def __init__(self, parent):
            self.parent = parent
            self.n_groups = parent.n_groups
            self.group_size = parent.group_size
            self.d = parent.d
            self.sse = 0.0
            self.n = 0
            self.active = 0.0
            self.alive = np.zeros(parent.n_groups, bool)

        def eval(self):
            self.parent.eval()
            return self

        def atoms(self):
            return self.parent.atoms()

        def encode(self, x):
            z = self.parent.encode(x)
            self.n += len(x)
            if not reconstruction:
                return z
            joint = (
                self.parent.solve(x).reshape(-1, self.n_groups, self.group_size)
                if isinstance(self.parent, NativeSLiCS)
                and self.parent.readout == "partial"
                else z
            )
            prediction = joint.flatten(1) @ self.parent.atoms().reshape(-1, self.d)
            self.sse += float((prediction - x).double().square().sum())
            active = joint.norm(dim=-1) > 1e-6
            self.active += float(active.sum())
            self.alive |= active.any(0).cpu().numpy()
            return z

    meter = Meter(m)
    ids = sorted({t[k] for t in targets for k in ["contrast_group", "detector_group"]})
    index = {g: i for i, g in enumerate(ids)}
    mapped = [
        dict(
            t,
            contrast_group=index[t["contrast_group"]],
            detector_group=index[t["detector_group"]],
        )
        for t in targets
    ]
    arrays = metrics.audit_arrays(
        ReadoutView(meter, ids), pack["x"], pack["masks"], pack["ids"], cfg, device
    )
    result = core.category_scores(arrays, mapped, cfg)
    if meter.n != len(pack["ids"]) * 196:
        raise ValueError("Every held-out patch must be scored")
    result["reconstruction"] = (
        dict(
            status="computed",
            mse=meter.sse / (meter.n * m.d),
            ve=1 - meter.sse / pack_variance(pack),
            active_groups=meter.active / meter.n,
            dead_groups=int((~meter.alive).sum()),
            patches=meter.n,
        )
        if reconstruction
        else dict(
            status="deferred",
            mse=None,
            ve=None,
            active_groups=None,
            dead_groups=None,
            patches=meter.n,
        )
    )
    for r, t in zip(result["categories"], targets):
        r.update(contrast_group=t["contrast_group"], detector_group=t["detector_group"])
    checked(result, cfg["category_indices"])
    if compact:
        result["_compact"] = study.compact_audit(arrays, matched.mapping(mapped), cfg)
    del arrays
    gc.collect()
    return result


# Same strict certificate and precision recovery as the completed COCO study.
partial_nnls = PrecisionRetry(partial_nnls)
_joint_retry = JointPrecisionRetry(Solver.solve)


def _solve(self, x, allowed=None, force_fallback=False):
    return _joint_retry(self, x, allowed, force_fallback=force_fallback)


Solver.solve = _solve

"""VOC-specific policy and resumable evaluation around the unchanged fit kernels."""


def protocol():
    return dict(
        version=1,
        seeds=[21, 22, 23],
        groups=19,
        width=12,
        top_k=8,
        top_k_rule="round(19*32/75), preserving original active-group fraction; fixed before VOC scoring",
        original_COCO_groups=75,
        original_top_k=32,
        GGF_LR=0.001,
        cap=0.6,
        warm_updates=7640,
        guided_updates=3000,
        SLiCS_updates=10640,
        images_per_update=8,
        GGF_LR_source="original COCO development-selected LR; no new VOC rate search",
        selection="Primary: completed endpoints; secondary: original development capture+0.2AP checkpoint rule, including update zero",
        supervision="same VOC FIT masks, one fixed native group per eligible category, common train-mask SVD directions",
        inference="GGF gated encoder; SLiCS independent 12-atom NNLS for object metrics, full joint NNLS for reconstruction",
        solver=dict(
            nnls_iterations=2000, nnls_tolerance=1e-5, nnls_fallback_iterations=20000
        ),
        split_scope="Existing VOC train/development/validation allocation, previously inspected; new model fits, not a fresh unseen dataset",
        compute="Time and allocated/reserved peaks; registered executed FLOPs and missing-operator inventory; no universal full-FLOP claim",
        no_old_checkpoint_writes=True,
    )


def fixed_targets(ctx, groups=None):
    if len(ctx["cfg"]["category_indices"]) != G:
        raise ValueError("VOC category denominator changed")
    return [
        dict(
            category_index=c,
            category=ctx["provenance"]["categories"][c]["name"],
            contrast_group=j,
            detector_group=j,
            heat_mean=0.0,
            heat_scale=1.0,
        )
        for j, c in enumerate(ctx["cfg"]["category_indices"])
    ]


@torch.no_grad()
def native_step(ctx, D, ids):
    p = ctx["packs"]["train"]
    xx = torch.as_tensor(
        np.array(p["x"][ids], copy=True), device=ctx["device"]
    ).reshape(-1, 768)
    allow, known = support(
        [p["masks"][i] for i in ids],
        ctx["cfg"]["category_indices"],
        dict(rank=12),
        "mask",
        ctx["device"],
    )
    take = torch.as_tensor(known, device=ctx["device"])
    xx = xx[take]
    allow = allow[take]
    D = D.clone()
    cols = allow.any(0)
    z = torch.zeros((len(xx), len(D)), device=ctx["device"])
    info = dict(cpu_fallback_rows=0, max_relative_kkt=0.0)
    if bool(cols.any()):
        zz, info = Solver(D[cols], SOLVER).solve(xx, allow[:, cols])
        z = z.to(zz.dtype)
        z[:, cols] = zz
    changed, atom = _precision_atom(atom_sweep, xx, D, z, allow, False)
    return changed, dict(solver=info, atom=atom)


def evaluate_state(ctx, row, pack, compact=False):
    st = memory.required(ctx["store"], row["key"])
    if row["method"] == "native_slics":
        assert training.weight_sha({"D": st["D"]}) == row["weights_sha"]
        m = NativeSLiCS(st["D"].to(ctx["device"]), SOLVER, "partial")
        out = evaluate_model(
            m,
            pack,
            row["targets"],
            ctx["cfg"],
            ctx["device"],
            compact=compact,
            reconstruction=False,
        )
    else:
        assert training.weight_sha(st["weights"]) == row["weights_sha"]
        m = model("released_vanilla", G, 12, K, 768).to(ctx["device"])
        m.load_state_dict(st["weights"])
        out = evaluate_model(
            m,
            pack,
            row["targets"],
            ctx["cfg"],
            ctx["device"],
            compact=compact,
            reconstruction=True,
        )
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def joint_evaluation(ctx, row, pack):
    from gpu_joint_nnls import GPUJointModel, benchmark as numerical_probe

    key = "joint_VOC_" + row["weights_sha"][:24]
    cached = ctx["store"].load(key)
    if cached and cached.get("complete"):
        return cached["report"]
    D = memory.required(ctx["store"], row["key"])["D"].to(ctx["device"])
    model = GPUJointModel(D, SOLVER)
    probe = numerical_probe(
        model, torch.as_tensor(np.array(pack["x"][0], copy=True), device=ctx["device"])
    )
    ctx["store"].write("joint_probe_" + row["weights_sha"][:16] + ".json", probe)
    st = cached or dict(
        images=0,
        sse=0.0,
        patches=0,
        seconds=0.0,
        signature=identity(
            dict(weights=row["weights_sha"], cohort=pack["identity"], solver=SOLVER)
        ),
    )
    assert st["signature"] == identity(
        dict(weights=row["weights_sha"], cohort=pack["identity"], solver=SOLVER)
    )
    try:
        for i in range(st["images"], len(pack["ids"])):
            start = time.monotonic()
            x = torch.as_tensor(np.array(pack["x"][i], copy=True), device=ctx["device"])
            z = model.solve(x)
            pred = z @ D.to(z.dtype)
            st.update(
                images=i + 1,
                sse=st["sse"] + float((pred - x).double().square().sum()),
                patches=st["patches"] + len(x),
                seconds=st["seconds"] + time.monotonic() - start,
            )
            if (i + 1) % 10 == 0 or i + 1 == len(pack["ids"]):
                ctx["store"].save(key, st)
                print(
                    "VOC joint reconstruction",
                    row["seed"],
                    i + 1,
                    "/",
                    len(pack["ids"]),
                    flush=True,
                )
    except BaseException:
        ctx["store"].save(key, st)
        raise
    report = dict(
        status="computed",
        mse=st["sse"] / (st["patches"] * 768),
        ve=1 - st["sse"] / pack_variance(pack),
        patches=st["patches"],
        seconds=st["seconds"],
    )
    st.update(complete=True, report=report)
    ctx["store"].save(key, st)
    return report


def final_row(ctx, seed, kind):
    if kind == "GGF":
        chosen = small_fit(ctx, seed, 0.001)
        st = memory.required(ctx["store"], chosen["fitted_key"])
        row = dict(
            chosen,
            key=chosen["fitted_key"],
            arm="GGF",
            method="released_vanilla",
            groups=G,
            active_groups=K,
            lr=0.001,
            cap=0.6,
            weights_sha=training.weight_sha(st["weights"]),
            selected_total_updates=TOTAL,
            training_seconds=memory.required(ctx["store"], f"smallG_s{seed}_warm")[
                "seconds"
            ]
            + st["seconds"],
        )
    else:
        native_fit(ctx, seed)
        key = f"native_slics12_s{seed}"
        st = memory.required(ctx["store"], key)
        row = dict(
            key=key,
            seed=seed,
            arm="SLiCS",
            method="native_slics",
            groups=G,
            rank=12,
            active_groups=None,
            weights_sha=training.weight_sha({"D": st["D"]}),
            targets=fixed_targets(ctx),
            solver=SOLVER,
            readout="partial",
            training_seconds=st["seconds"],
            selected_total_updates=TOTAL,
        )
    if st["updates"] != (GUIDED if kind == "GGF" else TOTAL):
        raise ValueError("Incomplete endpoint is not a final score")
    row["learned_parameters"] = (
        (2 * G * 12 * 768 + G * 12 + 2 * G + 2) if kind == "GGF" else G * 12 * 768
    )
    return row


def save_tables(ctx, records):
    import pandas as pd
    from scipy.stats import t

    flat = []
    for r in records:
        z = r["evaluation"]
        flat.append(
            dict(
                seed=r["seed"],
                method=r["arm"],
                Capture=100 * z["macro_capture"],
                patch_AP=100 * z["macro_ap"],
                VE=100 * z["reconstruction"]["ve"],
                training_loop_minutes=r["training_seconds"] / 60,
                learned_parameters=r["learned_parameters"],
                fit_invocation_seconds=r["fit_invocation_seconds"],
                fit_peak_allocated_GiB=r["fit_peak_allocated"] / 1024**3,
                fit_peak_reserved_GiB=r["fit_peak_reserved"] / 1024**3,
            )
        )
    frame = pd.DataFrame(flat)
    frame.to_csv(ctx["store"].results / "VOC_completed_per_seed.csv", index=False)
    frame.groupby("method").agg(
        {
            k: ["mean", "std"]
            for k in ["Capture", "patch_AP", "VE", "training_loop_minutes"]
        }
    ).to_csv(ctx["store"].results / "VOC_completed_mean_SD.csv")
    if len(records) == 6:
        paired = []
        for metric in ["Capture", "patch_AP", "VE"]:
            pivot = frame.pivot(index="seed", columns="method", values=metric)
            delta = pivot["GGF"] - pivot["SLiCS"]
            se = delta.std(ddof=1) / np.sqrt(len(delta))
            margin = t.ppf(0.975, len(delta) - 1) * se
            paired.append(
                dict(
                    metric=metric,
                    GGF_minus_SLiCS=float(delta.mean()),
                    sample_SD=float(delta.std(ddof=1)),
                    CI95_low=float(delta.mean() - margin),
                    CI95_high=float(delta.mean() + margin),
                )
            )
        pd.DataFrame(paired).to_csv(
            ctx["store"].results / "VOC_paired_intervals.csv", index=False
        )
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 3, figsize=(8, 2.8), layout="constrained")
        for ax, metric in zip(axs, ["Capture", "patch_AP", "VE"]):
            q = frame.groupby("method")[metric]
            ax.bar(
                ["GGF", "SLiCS"],
                q.mean().reindex(["GGF", "SLiCS"]),
                yerr=q.std().reindex(["GGF", "SLiCS"]),
                color=["#008879", "#9987b8"],
                capsize=4,
            )
            ax.set_title(metric)
            ax.set_ylabel("%")
            ax.spines[["top", "right"]].set_visible(False)
        fig.savefig(ctx["store"].results / "VOC_SLiCS_GGF.pdf")
        fig.savefig(ctx["store"].results / "VOC_SLiCS_GGF.png", dpi=200)
        plt.close(fig)
    print(frame.to_string(index=False), flush=True)


def profile_all(ctx, rows):
    from compute_probe import benchmark, bsf_factory, native_factory, concept_factory
    from voc_inputs import unique_manifest
    from coco_coordinated.run import FrozenStore

    profiles = []
    ids = list(range(8))
    s = ctx["store"]

    def run(name, factory, n, **facts):
        key = "profile_" + name
        old = s.load(key)
        if old is None:
            print("PROFILE", name, "disposable copies only", flush=True)
            old = dict(name=name, **facts, **benchmark(factory, n))
            s.save(key, old)
        profiles.append(old)
        s.write("compute_profiles.json", profiles)

    for row in [r for r in rows if r["seed"] == 21]:
        state = memory.required(s, row["key"])
        if row["arm"] == "GGF":
            for phase, train, adapt in [
                ("inference", False, False),
                ("training_plain", True, False),
                ("training_controller", True, True),
            ]:
                run(
                    "GGF19_" + phase,
                    bsf_factory(ctx, row, state, ids, train, adapt),
                    8,
                    method="GGF19",
                    phase=phase,
                    learned_parameters=row["learned_parameters"],
                )
        else:
            for phase in ["partial", "joint", "training"]:
                run(
                    "SLiCS19_" + phase,
                    native_factory(ctx, state["D"], ids, phase),
                    8,
                    method="SLiCS19",
                    phase=phase,
                    learned_parameters=row["learned_parameters"],
                    external_work="SciPy active-set NNLS and torch eigh have no complete registered FLOP formula",
                )
    primary = memory.required(ctx["source"], "primary_complete")["models"]
    for row in [r for r in primary if r["seed"] == 21]:
        state = memory.required(ctx["source"], row["key"])
        kind = row["kind"]
        rr = dict(row, groups=1024, active_groups=16, cap=0.4, lr=0.0003)
        for phase, train, adapt in [
            ("inference", False, False),
            ("training_plain", True, False),
        ] + ([("training_controller", True, True)] if kind == "agf_g" else []):
            run(
                kind + "1024_" + phase,
                bsf_factory(ctx, rr, state, ids, train, adapt),
                8,
                method=kind + "1024",
                phase=phase,
                learned_parameters=row["total_learned_parameters"],
            )
    root = Path(ctx["options"]["saved_root"]) / "voc_training/concept_sae_protocol_v2"
    path, manifest = unique_manifest(
        root,
        lambda c: (
            c.get("cohort") == "voc"
            and c.get("train") == ctx["packs"]["train"]["identity"]
            and c.get("normalization") == identity(ctx["norm"])
        ),
    )
    cs = FrozenStore(
        root / "checkpoints" / manifest["run_id"][:20], ctx["local"] / "read_concept"
    )
    state = memory.required(cs, "concept_sae_final_s21")
    if training.weight_sha(state["weights"]) != state["weights_sha"]:
        raise ValueError("Concept-SAE weight identity changed")
    run(
        "ConceptSAE_inference",
        concept_factory(ctx, state, ids),
        8,
        method="Concept-SAE",
        phase="inference",
        learned_parameters=state["parameters"],
    )
    for stage in [1, 2, 3]:
        run(
            "ConceptSAE_training_stage" + str(stage),
            concept_factory(ctx, state, list(range(64)), stage),
            64,
            method="Concept-SAE",
            phase="training_stage" + str(stage),
            learned_parameters=state["parameters"],
            optimizer_scope="Fresh Adam moments on disposable completed-model copy; architecture/stage objective/effective batch preserved",
        )
    import pandas as pd

    out = []
    for p in profiles:
        flop = p["flop_probes"][0]["registered_operator_FLOPs"]
        out.append(
            dict(
                name=p["name"],
                images=p["images"],
                learned_parameters=p["learned_parameters"],
                seconds_mean=p["seconds_mean"],
                seconds_SD=p["seconds_SD"],
                allocated_GiB=p["peak_GPU_allocated_bytes"] / 1024**3,
                reserved_GiB=p["peak_GPU_reserved_bytes"] / 1024**3,
                registered_FLOPs=flop,
                FLOPs_per_image=flop / p["images"],
                full_FLOPs_available=False,
            )
        )
    pd.DataFrame(out).to_csv(s.results / "compute_profiles.csv", index=False)
    s.write(
        "FLOP_COVERAGE.json",
        dict(
            full_FLOPs_available=False,
            covered="Registered matrix/convolution formulas executed inside forward, objectives, backward, Adam/controller and NNLS scopes",
            uncovered="Unregistered elementwise/reduction/indexing/eigen operations and external SciPy/BLAS CPU work; see per-operator inventory",
            do_not_infer="No full training FLOP or end-to-end runtime ratios from this partial count; cold-update repeated benchmarks are separate from historical fit timers",
        ),
    )


def main(o):
    import argparse, platform, zipfile
    from voc_inputs import prepare

    if not torch.cuda.is_available():
        raise RuntimeError("Select T4 or A100 for this notebook")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = prepare(o)
    s = ctx["store"]
    rows = []
    s.write(
        "hardware_invocation_" + str(time.time_ns()) + ".json",
        dict(
            GPU=torch.cuda.get_device_name(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            python=platform.python_version(),
            TF32=False,
        ),
    )
    # Freeze the transfer recipe before any newly fitted VOC endpoint is scored.
    s.save(
        "protocol_frozen_before_new_fits",
        dict(protocol=protocol(), selection_uses_VOC_validation=False),
    )
    for seed in SEEDS:
        for kind in ["GGF", "SLiCS"]:
            receipt = "completed_" + kind + "_" + str(seed)
            old = s.load(receipt)
            if old is None:
                print("START VOC", kind, "seed", seed, flush=True)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                try:
                    row = final_row(ctx, seed, kind)
                finally:
                    invocation = dict(
                        seed=seed,
                        method=kind,
                        seconds=time.monotonic() - start,
                        allocated=torch.cuda.max_memory_allocated(),
                        reserved=torch.cuda.max_memory_reserved(),
                        scope="This invocation: initialization, fitting and development validation; completed fit-loop timers are separate; resumed invocations accumulated below",
                    )
                    s.write(
                        "fit_invocations/"
                        + kind
                        + "_"
                        + str(seed)
                        + "_"
                        + str(time.time_ns())
                        + ".json",
                        invocation,
                    )
                history = [
                    json.loads(p.read_text())
                    for p in (s.results / "fit_invocations").glob(
                        kind + "_" + str(seed) + "_*.json"
                    )
                ]
                row.update(
                    fit_invocation_seconds=sum(p["seconds"] for p in history),
                    fit_peak_allocated=max(p["allocated"] for p in history),
                    fit_peak_reserved=max(p["reserved"] for p in history),
                )
                s.save("frozen_" + kind + "_" + str(seed), row)
                old = row
                old["evaluation"] = evaluate_state(ctx, row, ctx["packs"]["eval"], True)
                if kind == "SLiCS":
                    old["evaluation"]["reconstruction"] = joint_evaluation(
                        ctx, row, ctx["packs"]["eval"]
                    )
                s.save(receipt, old)
            rows.append(old)
            save_tables(ctx, rows)
        print("COMPLETED VOC PAIR", seed, flush=True)
        if seed == 21:
            profile_all(ctx, rows)
    profile_all(ctx, rows)
    for stamp in ctx["source_stamps"]:
        assert sha_file(stamp["path"]) == stamp["sha"], (
            "A frozen source manifest changed"
        )
    s.write(
        "completion.json",
        dict(
            complete=True,
            primary_final_endpoints=True,
            pairs=len(SEEDS),
            original_results_preserved=True,
            full_FLOPs_available=False,
        ),
    )
    archive = s.results.parent / "VOC_SLiCS_GGF_compute_results.zip"
    pending = archive.with_suffix(".pending.zip")
    with zipfile.ZipFile(pending, "w", zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(s.results))
    with zipfile.ZipFile(pending) as z:
        assert z.testzip() is None
    os.replace(pending, archive)
    Path(o["receipt"]).write_text(
        json.dumps(dict(complete=True, results=str(s.results), archive=str(archive)))
    )
    print("WHOLE NOTEBOOK COMPLETE", archive, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(json.loads(Path(args.config).read_text()))
