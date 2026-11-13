"""Predeclared capacity-matched DINOv3 task adaptation of SLiCS Algorithm 1.

Not a replication of SLiCS's CLIP/global-image retrieval protocol. Sources and
historical models stay read-only. Test is the previously inspected GGF holdout.
"""

import copy
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from coco_adaptation import core, inputs, coverage
from coco_coordinated import run as memory
from . import slics_comparison as old, study


def settings():
    return dict(
        version=1,
        study="ggf_slics_closeout",
        seeds=[21, 22, 23],
        groups=75,
        rank=3,
        active_groups=16,
        images_per_update=4,
        warm_updates=2500,
        guided_updates=2500,
        slics_epochs=10,
        evaluate_every=500,
        save_every=50,
        log_every=25,
        nnls_iterations=500,
        nnls_tolerance=1e-4,
        lr=0.0003,
        snr=0.1,
        lambda_capture=0.05,
        lambda_detection=0.01,
        bootstrap=200,
        include_dense_first_seed=True,
        atom_nonincrease_guard=False,
        max_relative_mse_growth=0.02,
        max_ve_cost=0.01,
        scope="Capacity-matched task adaptation; reused inspected holdout.",
    )


def validate(cfg):
    if set(cfg) != set(settings()):
        raise ValueError("Unknown or missing settings")
    if cfg["seeds"] != [21, 22, 23]:
        raise ValueError("Report all three predeclared starts")
    if (cfg["groups"], cfg["rank"], cfg["active_groups"]) != (75, 3, 16):
        raise ValueError("Capacity changed")
    for k in (
        "images_per_update",
        "warm_updates",
        "guided_updates",
        "slics_epochs",
        "evaluate_every",
        "save_every",
        "log_every",
        "nnls_iterations",
        "bootstrap",
    ):
        if type(cfg[k]) is not int or cfg[k] <= 0:
            raise ValueError("Positive integer: " + k)
    if (cfg["lr"], cfg["snr"], cfg["lambda_capture"], cfg["lambda_detection"]) != (
        0.0003,
        0.1,
        0.05,
        0.01,
    ):
        raise ValueError("Keep the frozen GGF loss recipe")
    if not 0 < cfg["nnls_tolerance"] < 1:
        raise ValueError("NNLS tolerance")
    if cfg["max_relative_mse_growth"] != 0.02 or cfg["max_ve_cost"] != 0.01:
        raise ValueError("Predeclared guards")
    if (
        type(cfg["include_dense_first_seed"]) is not bool
        or type(cfg["atom_nonincrease_guard"]) is not bool
    ):
        raise ValueError("Boolean flags")
    return cfg


def prepare(
    source_output, output, local, source_sha, versions, cfg, device, require_drive=True
):
    cfg = validate(copy.deepcopy(cfg))
    legacy = old.settings()
    legacy["seeds"] = cfg["seeds"]
    ctx = old.prepare(
        source_output,
        Path(output) / "input_verification",
        local,
        source_sha,
        versions,
        legacy,
        device,
        require_drive,
    )
    ctx["comparison"] = cfg
    ctx["cfg"].update(
        updates=cfg["guided_updates"],
        evaluate_every=cfg["evaluate_every"],
        local_save_every=cfg["save_every"],
        diagnostic_grad_every=cfg["evaluate_every"],
        bootstrap=cfg["bootstrap"],
        images_per_update=cfg["images_per_update"],
        audit_image_batch=2,
        batch=512,
        min_category_images=5,
        max_audit_groups=4,
    )
    ctx["store"] = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=source_sha,
            versions=versions,
            device=device,
            source_run=ctx["source"].run_id,
            activation_run=ctx["original"].run_id,
        ),
        Path(local) / "closeout_rescue",
        require_drive,
    )
    n = len(ctx["packs"]["train"]["ids"])
    if n % cfg["images_per_update"]:
        raise ValueError("Full image epochs require divisible batch size")
    if cfg["warm_updates"] + cfg["guided_updates"] != cfg["slics_epochs"] * (
        n // cfg["images_per_update"]
    ):
        raise ValueError("SLiCS/SGD total image presentations must match")
    if cfg["warm_updates"] % (n // cfg["images_per_update"]):
        raise ValueError("Warm-up must end on an epoch boundary")
    if len(ctx["cfg"]["category_indices"]) != cfg["groups"]:
        raise ValueError("Frozen cohort must have 75 categories")
    # Check availability/identity, without opening test activations or labels.
    plan = memory.required(ctx["source"], "fresh_plan")
    test_ids = plan["splits"]["test"]
    used = set(i for ids in ctx["provenance"]["splits"].values() for i in ids)
    if (
        len(test_ids) != 1500
        or len(set(test_ids)) != 1500
        or used.intersection(test_ids)
    ):
        raise ValueError("Saved holdout identity/overlap")
    if plan["coverage"]["category_indices"] != ctx["cfg"]["category_indices"]:
        raise ValueError("Holdout cohort differs")
    for lo in range(0, len(test_ids), ctx["source"].config["config"]["extract_chunk"]):
        if not ctx["source"]._records(ctx["source"].root, f"features_test_{lo:06d}"):
            raise FileNotFoundError(
                "Missing saved test shard; no extraction/retraining fallback"
            )
    ctx["store"].write(
        "protocol_settings.json",
        dict(
            config=cfg,
            base=ctx["base"],
            fit_recipe=ctx["cfg"],
            train_ids_sha=identity(ctx["packs"]["train"]["ids"]),
            development_ids_sha=identity(ctx["packs"]["select"]["ids"]),
            test_ids_sha=identity(test_ids),
            train_images=n,
            selection_images=len(ctx["packs"]["select"]["ids"]),
            test_images=1500,
            holdout="Previously inspected GGF 1500-image holdout, disjoint from train/development/calibration. Not a new independent confirmation.",
            supervision="Mask-support SLiCS and GGF see the same training annotations. Image-presence SLiCS is a weaker-supervision sensitivity control.",
            inference="No labels/masks/categories enter encode(). Shared Top-16 readout is primary. Dense NNLS is separate first-seed sensitivity.",
            compute="Same ordered image batches and 10 training passes. NNLS/SVD sweeps and gradient updates are NOT equal compute/parameters.",
            paper_differences=[
                "DINOv3 patches, not original CLIP/global-image retrieval",
                "75 object categories, rank 3, frozen GGF train-only normalization",
                "4-image minibatches, finite NNLS with KKT residuals",
                "mask support plus image-presence control; Top-16 prune/refit is an inference adaptation",
                "Algorithm 1 nonzero coefficient support and SVD/sign/clamp; practical no-guard atom updates by default",
                "Validation checkpoint choice; final checkpoints retained; historical test reuse disclosed",
            ],
            checkpoint_rule="Same capture + 0.2 AP score, within 2% relative MSE of common native start. Preserve all curves and failures.",
            excluded="No full-capacity 1024-group score in matched table; no structural edits, Gaussian/EM, scalar bank or new baseline retraining.",
        ),
    )
    return ctx


def batch(ctx, seed, u):
    n = len(ctx["packs"]["train"]["ids"])
    b = ctx["comparison"]["images_per_update"]
    nb = n // b
    epoch, part = divmod(u, nb)
    perm = np.random.default_rng(
        np.random.SeedSequence([10606, seed, epoch])
    ).permutation(n)
    ids = perm[part * b : (part + 1) * b]
    p = ctx["packs"]["train"]
    return (
        torch.as_tensor(np.array(p["x"][ids], copy=True), device=ctx["device"]),
        [p["masks"][i] for i in ids],
        identity(ids),
    )


def support(masks, cohort, cfg, kind, device):
    f = np.stack([m["fraction"] for m in masks]).astype(np.float32) / 255
    known = np.stack([m["crowd"] for m in masks]).reshape(-1) <= 0.1
    if kind == "mask":
        pos = f[:, :, cohort] >= 0.5
    elif kind == "image":
        present = np.array(
            [
                [any(o["category"] == c for o in m["instances"]) for c in cohort]
                for m in masks
            ]
        )
        pos = np.repeat(present[:, None, :], 196, axis=1)
    else:
        raise ValueError("Support kind")
    return torch.as_tensor(
        np.repeat(pos.reshape(-1, len(cohort)), cfg["rank"], axis=1), device=device
    ), known


@torch.no_grad()
def solve(x, D, allowed=None, iterations=500, tolerance=1e-4):
    """Exactly remove columns forbidden for EVERY row before solving NNLS."""
    if allowed is None:
        return old.nnls(x, D, iterations=iterations, tolerance=tolerance)
    columns = allowed.any(0)
    out = torch.zeros((len(x), len(D)), dtype=x.dtype, device=x.device)
    if not bool(columns.any()):
        return out, dict(
            iterations=0,
            relative_kkt_error=0.0,
            converged=True,
            inactive_code_max=0.0,
            solver_atoms=0,
        )
    z, info = old.nnls(x, D[columns], allowed[:, columns], iterations, tolerance)
    out[:, columns] = z
    info["solver_atoms"] = int(columns.sum())
    return out, info


@torch.no_grad()
def atom_sweep(x, D, z, allowed, guard=False):
    """Paper Algorithm 1: update only currently NONZERO coefficients, not all allowed rows."""
    if bool((z[~allowed] != 0).any()):
        raise ValueError("Unsupported coefficients")
    residual = x - z @ D
    before = float(residual.square().sum())
    accepted = rejected = dead = 0
    for j in range(len(D)):
        rows = z[:, j] > 0
        if not bool(rows.any()):
            dead += 1
            continue
        E = residual[rows] + z[rows, j, None] * D[j]
        if float(E.square().sum()) <= 1e-12:
            dead += 1
            continue
        if len(E) <= E.shape[1]:
            _, v = torch.linalg.eigh(E @ E.T)
            atom = E.T @ v[:, -1]
            atom /= atom.norm().clamp_min(1e-12)
        else:
            _, v = torch.linalg.eigh(E.T @ E)
            atom = v[:, -1]
        beta = E @ atom
        if float(beta.clamp_min(0).square().sum()) < float(
            (-beta).clamp_min(0).square().sum()
        ):
            atom = -atom
            beta = -beta
        beta = beta.clamp_min(0)
        new = E - beta[:, None] * atom
        if guard and float(new.square().sum()) > float(
            residual[rows].square().sum()
        ) + 1e-7 * max(1.0, before):
            rejected += 1
            continue
        D[j] = atom
        z[rows, j] = beta
        residual[rows] = new
        accepted += 1
    if not torch.isfinite(D).all():
        raise FloatingPointError("SVD dictionary")
    return D, dict(
        atom_updates=accepted,
        rejected_atom_updates=rejected,
        unused_atoms=dead,
        residual_sse_before=before,
        residual_sse_after=float(residual.square().sum()),
        support="nonzero coefficients",
        guard=guard,
    )


class SlicsModel(old.SlicsModel):
    def __init__(self, D, cfg, dense=False):
        if len(D) != cfg["groups"] * cfg["rank"]:
            raise ValueError("Dictionary atom count")
        super().__init__(D, cfg, dense)
        self.solver_totals = dict(
            calls=0, nonconverged=0, max_relative_kkt_error=0.0, total_iterations=0
        )

    def encode(self, x):
        started = time.monotonic()
        z, first = solve(
            x,
            self.D,
            iterations=self.cfg["nnls_iterations"],
            tolerance=self.cfg["nnls_tolerance"],
        )
        stages = [dict(stage="dense", **first)]
        if not self.dense:
            heat = z.reshape(-1, self.n_groups, self.group_size).norm(dim=-1)
            ids = heat.topk(
                min(self.cfg["active_groups"], self.n_groups), dim=-1
            ).indices
            on = torch.zeros_like(heat, dtype=torch.bool).scatter_(1, ids, True)
            z, second = solve(
                x,
                self.D,
                on.repeat_interleave(self.group_size, dim=1),
                self.cfg["nnls_iterations"],
                self.cfg["nnls_tolerance"],
            )
            stages.append(dict(stage="restricted", **second))
        for info in stages:
            t = self.solver_totals
            t["calls"] += 1
            t["nonconverged"] += int(not info["converged"])
            t["max_relative_kkt_error"] = max(
                t["max_relative_kkt_error"], info["relative_kkt_error"]
            )
            t["total_iterations"] += info["iterations"]
        self.solver_diagnostics.extend(stages)
        if len(self.solver_diagnostics) > 100:
            del self.solver_diagnostics[:-100]
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        self.inference_seconds += time.monotonic() - started
        self.inference_patches += len(x)
        return z.reshape(-1, self.n_groups, self.group_size)


@torch.no_grad()
def init_slics(ctx, seed, kind):
    """All training-supported patches, streamed Gram SVD; exactly 75 groups."""
    key = f"svd_{kind}_s{seed}"
    s = ctx["store"]
    old_state = s.load(key)
    if old_state is not None:
        return old_state["D"].to(ctx["device"])
    started = time.monotonic()
    p = ctx["packs"]["train"]
    atoms = []
    report = []
    for g, c in enumerate(ctx["cfg"]["category_indices"]):
        gram = np.zeros((p["x"].shape[-1],) * 2, np.float64)
        count = 0
        for i, m in enumerate(p["masks"]):
            known = m["crowd"] <= 0.1
            take = known & (
                (m["fraction"][:, c].astype(float) / 255 >= 0.5)
                if kind == "mask"
                else any(o["category"] == c for o in m["instances"])
            )
            E = np.asarray(p["x"][i][take], np.float64)
            if len(E):
                gram += E.T @ E
                count += len(E)
        if count < ctx["comparison"]["rank"]:
            raise ValueError("Too few training patches for SVD category " + str(c))
        eig, v = np.linalg.eigh(gram)
        D = v[:, -ctx["comparison"]["rank"] :].T[::-1].copy()
        pos = np.zeros(len(D))
        neg = pos.copy()
        for i, m in enumerate(p["masks"]):
            known = m["crowd"] <= 0.1
            take = known & (
                (m["fraction"][:, c].astype(float) / 255 >= 0.5)
                if kind == "mask"
                else any(o["category"] == c for o in m["instances"])
            )
            beta = np.asarray(p["x"][i][take], np.float64) @ D.T
            pos += np.square(np.maximum(beta, 0)).sum(0)
            neg += np.square(np.minimum(beta, 0)).sum(0)
        D[pos < neg] *= -1
        atoms.append(D.astype(np.float32))
        report.append(
            dict(
                category_index=c,
                group=g,
                patches=count,
                eigenvalues=eig[-len(D) :][::-1].tolist(),
            )
        )
        s.event(
            "svd_initialization",
            seed=seed,
            kind=kind,
            category=g + 1,
            total=len(ctx["cfg"]["category_indices"]),
            patches=count,
        )
    D = torch.as_tensor(np.concatenate(atoms), device=ctx["device"])
    s.save(key, dict(D=D.cpu(), report=report, seconds=time.monotonic() - started))
    s.write("initialization/" + key + ".json", report)
    return D


@contextmanager
def progress(ctx, m, label):
    original = m.encode
    status = dict(last=time.monotonic(), patches=0)

    def tracked(x):
        z = original(x)
        status["patches"] += len(x)
        if time.monotonic() - status["last"] >= 20:
            ctx["store"].event(
                "inference_progress", work=label, patches=status["patches"]
            )
            status["last"] = time.monotonic()
        return z

    m.encode = tracked
    try:
        yield
    finally:
        m.encode = original


def checked_dev(ctx, m, targets):
    ctx["store"].event("development_scoring_start", model=type(m).__name__)
    with progress(ctx, m, "development"):
        v = old.dev(ctx, m, targets)
    if len(v["categories"]) != 75 or any(
        r["capture"] is None or r["ap"] is None for r in v["categories"]
    ):
        raise ValueError(
            "Cannot silently reduce the 75-category evaluation denominator"
        )
    return v


def fit_sgd(
    ctx,
    seed,
    key,
    initial=None,
    targets=None,
    supervised=False,
    max_updates=None,
    warm_initial=None,
    reference_mse=None,
    observer=None,
):
    s = ctx["store"]
    cfg = ctx["comparison"]
    warm = initial is None
    total = cfg["warm_updates"] if warm else cfg["guided_updates"]
    torch.manual_seed(106060 + seed)
    m = training.make_model(
        "vanilla", ctx["base"], ctx["packs"]["train"]["x"].shape[-1]
    ).to(ctx["device"])
    if initial is not None:
        m.load_state_dict(initial)
    if warm_initial is not None:
        if not warm or supervised:
            raise ValueError("Warm initializer only for reconstruction warm-up")
        m.load_state_dict(warm_initial)
    start_sha = (
        None
        if warm and warm_initial is None
        else training.weight_sha(warm_initial if warm else initial)
    )
    head = torch.nn.Parameter(
        torch.zeros(0 if targets is None else len(targets), 2, device=ctx["device"])
    )
    opt = torch.optim.Adam(
        list(m.parameters()) + ([head] if supervised else []), lr=cfg["lr"]
    )
    st = s.load(key)

    def measure():
        if warm:
            r = training.measure(
                m, ctx["packs"]["select"]["x"].reshape(-1, m.d), ctx["device"], 512
            )
            return dict(reconstruction=r, score=-r["mse"])
        return checked_dev(ctx, m, targets)

    if st is None:
        v = measure()
        first_ok = warm or v["reconstruction"]["mse"] <= (
            v["reconstruction"]["mse"] if reference_mse is None else reference_mse
        ) * (1 + cfg["max_relative_mse_growth"])
        st = dict(
            updates=0,
            total=total,
            initial_sha=start_sha,
            supervised=supervised,
            target_sha=identity(targets),
            best=training.weights(m),
            best_update=0,
            best_score=v["score"] if first_ok else -1e100,
            initial_mse=v["reconstruction"]["mse"]
            if reference_mse is None
            else reference_mse,
            reference_mse=reference_mse,
            history=[dict(update=0, validation=v, eligible=bool(first_ok))],
            seconds=0.0,
            batch_sha=identity([]),
        )
    else:
        if (
            st["total"],
            st["initial_sha"],
            st["supervised"],
            st["target_sha"],
            st.get("reference_mse"),
        ) != (total, start_sha, supervised, identity(targets), reference_mse):
            raise ValueError("SGD resume identity")
        m.load_state_dict(st["weights"])
        head.data.copy_(st["head"].to(ctx["device"]))
        opt.load_state_dict(st["optimizer"])
        torch.set_rng_state(st["rng"])
        if ctx["device"] == "cuda":
            torch.cuda.set_rng_state_all(st["cuda_rng"])

    def save(drive=True):
        st.update(
            weights=training.weights(m),
            head=head.detach().cpu().clone(),
            optimizer=copy.deepcopy(opt.state_dict()),
            rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if ctx["device"] == "cuda"
            else None,
        )
        s.save(key, st, drive)

    stop = total if max_updates is None else min(total, max_updates)
    print(key, "resume", st["updates"], "/", total, flush=True)
    if observer is not None:
        observer(m, st["updates"], key)
    try:
        while st["updates"] < stop:
            u = st["updates"]
            x, masks, bh = batch(ctx, seed, u + (0 if warm else cfg["warm_updates"]))
            st["batch_sha"] = identity([st["batch_sha"], bh])
            m.train()
            start = time.monotonic()
            xx = x.flatten(0, 1)
            noisy = xx + cfg["snr"] * xx.std() * torch.randn_like(xx)
            native, info = m.loss(noisy, target=xx)
            cap = det = native * 0
            objects = 0
            if supervised:
                cap, det, objects = core.supervised_loss(
                    m, x, masks, targets, head, ctx["cfg"]
                )
            terms = dict(
                native=native,
                capture=cfg["lambda_capture"] * cap,
                detection=cfg["lambda_detection"] * det,
            )
            grads = (
                study.component_gradients(m, terms)
                if supervised and (u + 1) % cfg["evaluate_every"] == 0
                else None
            )
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss")
            opt.zero_grad()
            loss.backward()
            gn = float(
                torch.nn.utils.clip_grad_norm_(
                    list(m.parameters()) + ([head] if supervised else []), 5.0
                )
            )
            if not np.isfinite(gn):
                raise FloatingPointError("Gradient")
            opt.step()
            m.normalize_decoder()
            study.sync(ctx["device"])
            st["seconds"] += time.monotonic() - start
            st["updates"] += 1
            u = st["updates"]
            if u % cfg["evaluate_every"] == 0 or u == total:
                v = measure()
                eligible = warm or v["reconstruction"]["mse"] <= st["initial_mse"] * (
                    1 + cfg["max_relative_mse_growth"]
                )
                if eligible and v["score"] > st["best_score"]:
                    st.update(
                        best=training.weights(m), best_score=v["score"], best_update=u
                    )
                if observer is not None:
                    observer(m, u, key)
                st["history"].append(
                    dict(
                        update=u,
                        validation=v,
                        eligible=bool(eligible),
                        rejection_reasons=[] if eligible else ["reconstruction_guard"],
                        losses={k: float(t.detach()) for k, t in terms.items()},
                        gradient_norm=gn,
                        component_gradients=grads,
                        objects=objects,
                    )
                )
                s.write("histories/" + key + ".json", st["history"])
                s.event(
                    "development_checkpoint",
                    key=key,
                    update=u,
                    total=total,
                    eligible=eligible,
                    validation={k: x for k, x in v.items() if k != "categories"},
                )
            elif u % cfg["log_every"] == 0:
                s.event(
                    "training_progress",
                    key=key,
                    update=u,
                    total=total,
                    seconds_per_update=st["seconds"] / u,
                )
            if u % cfg["save_every"] == 0 or u == total:
                save()
        save()
    except BaseException:
        save(False)
        raise
    return st


def fit_slics(ctx, seed, kind, initial, initial_mse, max_updates=None):
    s = ctx["store"]
    cfg = ctx["comparison"]
    key = f"slics_{kind}_s{seed}"
    total = cfg["warm_updates"] + cfg["guided_updates"]
    st = s.load(key)
    if st is None:
        st = dict(
            updates=0,
            total=total,
            D=initial.cpu().clone(),
            best=None,
            best_score=None,
            best_update=None,
            initial_sha=identity(initial.cpu().numpy()),
            initial_mse=initial_mse,
            history=[],
            seconds=0.0,
            batch_sha=identity([]),
            solver_calls=0,
            nonconverged_batches=0,
            max_kkt=0.0,
        )
    if st["total"] != total or st["initial_sha"] != identity(initial.cpu().numpy()):
        raise ValueError("SLiCS resume identity")
    m = SlicsModel(st["D"].to(ctx["device"]), cfg)
    targets = old.sl_targets(ctx, m)

    def evaluate(u, solve=None, atom=None):
        v = checked_dev(ctx, m, targets)
        eligible = v["reconstruction"]["mse"] <= initial_mse * (
            1 + cfg["max_relative_mse_growth"]
        )
        if eligible and (st["best_score"] is None or v["score"] > st["best_score"]):
            st.update(
                best=m.D.detach().cpu().clone(), best_score=v["score"], best_update=u
            )
        st["history"].append(
            dict(
                update=u,
                validation=v,
                eligible=bool(eligible),
                rejection_reasons=[] if eligible else ["reconstruction_guard"],
                solver=solve,
                atom_sweep=atom,
            )
        )
        s.write("histories/" + key + ".json", st["history"])
        s.event(
            "development_checkpoint",
            key=key,
            update=u,
            total=total,
            eligible=eligible,
            validation={k: x for k, x in v.items() if k != "categories"},
            solver=solve,
        )

    def save(drive=True):
        st["D"] = m.D.detach().cpu().clone()
        s.save(key, st, drive)

    stop = total if max_updates is None else min(total, max_updates)
    print(key, "resume", st["updates"], "/", total, flush=True)
    try:
        if not st["history"]:
            evaluate(0)
            save()
        while st["updates"] < stop:
            x, masks, bh = batch(ctx, seed, st["updates"])
            allow, known = support(
                masks, ctx["cfg"]["category_indices"], cfg, kind, ctx["device"]
            )
            take = torch.as_tensor(known, device=ctx["device"])
            xx = x.flatten(0, 1)[take]
            allow = allow[take]
            if not len(xx):
                raise ValueError("All crowd batch")
            started = time.monotonic()
            z, solve_info = solve(
                xx, m.D, allow, cfg["nnls_iterations"], cfg["nnls_tolerance"]
            )
            D, atom = atom_sweep(xx, m.D, z, allow, cfg["atom_nonincrease_guard"])
            m.D.copy_(D)
            study.sync(ctx["device"])
            st["seconds"] += time.monotonic() - started
            st["updates"] += 1
            u = st["updates"]
            st["batch_sha"] = identity([st["batch_sha"], bh])
            st["solver_calls"] += 1
            st["nonconverged_batches"] += int(not solve_info["converged"])
            st["max_kkt"] = max(st["max_kkt"], solve_info["relative_kkt_error"])
            solve_info.update(
                unsupported_patch_fraction=float((~allow.any(1)).float().mean()),
                crowd_excluded_fraction=float(1 - known.mean()),
            )
            if u % cfg["evaluate_every"] == 0 or u == total:
                evaluate(u, solve_info, atom)
            elif u % cfg["log_every"] == 0:
                s.event(
                    "training_progress",
                    key=key,
                    update=u,
                    total=total,
                    solver=solve_info,
                    atom=atom,
                    seconds_per_update=st["seconds"] / u,
                )
            if u % cfg["save_every"] == 0 or u == total:
                save()
        save()
    except BaseException:
        save(False)
        raise
    return st


def model_hash(w):
    return (
        training.weight_sha(w)
        if isinstance(w, dict)
        else memory.streaming_array_sha(w.numpy())
    )


def train(ctx):
    s = ctx["store"]
    cfg = ctx["comparison"]
    frozen = s.load("comparison_frozen")
    if frozen is not None:
        return frozen
    rows = []
    for seed in cfg["seeds"]:
        s.event("seed_start", seed=seed)
        native = fit_sgd(ctx, seed, f"native_s{seed}")
        initial = native["best"]
        m = core.new_model("vanilla", ctx["base"], initial, ctx["device"])
        targets = s.load(f"targets_s{seed}")
        if targets is None:
            with progress(ctx, m, "training_group_assignment"):
                targets = s.save(f"targets_s{seed}", old.fixed_category_targets(ctx, m))
        mse = checked_dev(ctx, m, targets)["reconstruction"]["mse"]
        del m
        for arm, supervised in [("continued", False), ("ggf", True)]:
            key = f"{arm}_s{seed}"
            st = fit_sgd(ctx, seed, key, initial, targets, supervised)
            rows.append(
                dict(
                    key=key,
                    seed=seed,
                    arm=arm,
                    kind="native",
                    field="best",
                    weights_sha=model_hash(st["best"]),
                    best_update=st["best_update"],
                    eligible=True,
                    training_seconds=native["seconds"] + st["seconds"],
                    updates=cfg["warm_updates"] + st["updates"],
                    batch_sha=st["batch_sha"],
                    parameters=2 * 225 * 768 + 225 + (150 if supervised else 0),
                )
            )
        a, b = [s.load(f"{arm}_s{seed}") for arm in ("continued", "ggf")]
        if a["batch_sha"] != b["batch_sha"]:
            raise ValueError("Paired SGD batches differ")
        for kind in ("mask", "image"):
            D = init_slics(ctx, seed, kind)
            svdkey = f"svd_{kind}_s{seed}"
            init_seconds = s.load(svdkey)["seconds"]
            rows.append(
                dict(
                    key=svdkey,
                    seed=seed,
                    arm="SVD " + kind,
                    kind="slics",
                    field="D",
                    weights_sha=model_hash(D.cpu()),
                    eligible=None,
                    best_update=0,
                    updates=0,
                    training_seconds=init_seconds,
                    parameters=225 * 768,
                )
            )
            st = fit_slics(ctx, seed, kind, D, mse)
            field = "best" if st["best"] is not None else "D"
            rows.append(
                dict(
                    key=f"slics_{kind}_s{seed}",
                    seed=seed,
                    arm="SLiCS " + kind,
                    kind="slics",
                    field=field,
                    weights_sha=model_hash(st[field]),
                    eligible=st["best"] is not None,
                    best_update=st["best_update"],
                    updates=st["updates"],
                    training_seconds=st["seconds"] + init_seconds,
                    parameters=225 * 768,
                    solver_nonconverged_batches=st["nonconverged_batches"],
                    solver_max_kkt=st["max_kkt"],
                )
            )
        del native, initial, a, b
        gc.collect()
        if ctx["device"] == "cuda":
            torch.cuda.empty_cache()
    out = dict(
        complete=True,
        models=rows,
        test_used=False,
        groups=75,
        rank=3,
        active_groups=16,
        source_sha=s.config["source_sha"],
    )
    s.save("comparison_frozen", out)
    s.write("models_frozen.json", out)
    memory.assert_immutable(ctx["stamps"])
    return out


def load_audit_packs(ctx):
    if ctx["store"].load("comparison_frozen") is None:
        raise RuntimeError("Freeze EVERY model before opening audit")
    norm = memory.required(ctx["original"], "normalization")
    mapping = memory.normalized_pack(
        ctx["original"],
        "mapping",
        ctx["provenance"],
        norm,
        ctx["local"],
        ctx["cfg"]["extract_chunk"],
    )
    plan = memory.required(ctx["source"], "fresh_plan")
    test = memory.normalized_pack(
        ctx["source"],
        "test",
        dict(splits=plan["splits"]),
        norm,
        ctx["local"],
        ctx["source"].config["config"]["extract_chunk"],
    )
    sets = [
        set(p["ids"])
        for p in [ctx["packs"]["train"], ctx["packs"]["select"], mapping, test]
    ]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Audit split overlap")
    actual = coverage.evidence(test, ctx["cfg"]).all(2).sum(0)
    if not np.array_equal(actual, np.asarray(plan["coverage"]["split_counts"]["test"])):
        raise ValueError("Saved test coverage changed")
    ctx["store"].write(
        "audit_data_identity.json",
        dict(
            mapping_ids_sha=identity(mapping["ids"]),
            test_ids_sha=identity(test["ids"]),
            normalization_sha=identity(norm),
            test_coverage=actual.tolist(),
            no_overlap=True,
            test_previously_inspected=True,
        ),
    )
    return mapping, test


def evaluate(ctx):
    s = ctx["store"]
    cached = s.load("comparison_summary")
    if cached is not None:
        return cached
    frozen = memory.required(s, "comparison_frozen")
    mapping, test = load_audit_packs(ctx)
    reports = []
    for row in frozen["models"]:
        modes = [False]
        if (
            ctx["comparison"]["include_dense_first_seed"]
            and row["seed"] == ctx["comparison"]["seeds"][0]
            and row["arm"].startswith("SLiCS")
        ):
            modes += [True]
        for dense in modes:
            tag = row["key"] + ("_dense" if dense else "_top16")
            prior = s.load("report_" + tag)
            if prior is not None:
                reports.append(prior)
                continue
            s.event("audit_start", key=tag, images=len(test["ids"]))
            w = memory.required(s, row["key"])[row["field"]]
            if model_hash(w) != row["weights_sha"]:
                raise ValueError("Frozen weights changed")
            m = (
                SlicsModel(w.to(ctx["device"]), ctx["comparison"], dense)
                if row["kind"] == "slics"
                else core.new_model("vanilla", ctx["base"], w, ctx["device"])
            )
            maps = s.load("mapping_" + tag)
            if maps is None:
                s.event("calibration_start", key=tag, images=len(mapping["ids"]))
                with progress(ctx, m, tag + "_calibration"):
                    a = metrics.audit_arrays(
                        m,
                        mapping["x"],
                        mapping["masks"],
                        mapping["ids"],
                        ctx["cfg"],
                        ctx["device"],
                    )
                maps = metrics.calibrate(a, ctx["cfg"], ctx["provenance"]["categories"])
                del a
                gc.collect()
                s.save("mapping_" + tag, maps)
                s.write("mappings/" + tag + ".json", maps["mappings"])
            started = time.monotonic()
            with progress(ctx, m, tag + "_test"):
                audit = metrics.audit_arrays(
                    m, test["x"], test["masks"], test["ids"], ctx["cfg"], ctx["device"]
                )
            per = metrics.evaluate(audit, maps, dict(ctx["cfg"], bootstrap=0), 71002)
            indexed = {r["category_index"]: r for r in per}
            cohort = [indexed[c] for c in ctx["cfg"]["category_indices"]]
            if any(r["contrast"] is None or r["detection"] is None for r in cohort):
                raise ValueError("Missing test category; denominator cannot shrink")
            with progress(ctx, m, tag + "_reconstruction"):
                rec = training.measure(
                    m, test["x"].reshape(-1, m.d), ctx["device"], 512
                )
            if not dense and rec["active_groups"] > 16.00001:
                raise RuntimeError("Inference activity budget violated")
            compact = study.compact_audit(audit, maps, ctx["cfg"])
            s.save("compact_" + tag, compact)
            summary = dict(
                **row,
                readout="dense NNLS" if dense else "Top-16",
                images=len(test["ids"]),
                categories=len(cohort),
                capture=float(
                    np.mean([r["contrast"]["single_group_capture"] for r in cohort])
                ),
                ap=float(np.mean([r["detection"]["ap"] for r in cohort])),
                all_group_capture=float(
                    np.mean([r["contrast"]["all_group_capture"] for r in cohort])
                ),
                reconstruction=rec,
                audit_seconds=time.monotonic() - started,
                solver=getattr(m, "solver_totals", None),
                inference_seconds=getattr(m, "inference_seconds", None),
                inference_patches=getattr(m, "inference_patches", None),
            )
            result = dict(
                tag=tag,
                summary=summary,
                categories=per,
                skipped_objects=audit["skipped"],
            )
            s.save("report_" + tag, result)
            s.write("reports/" + tag + ".json", result)
            reports.append(result)
            s.event("audit_complete", **summary)
            del m, audit, compact, w
            gc.collect()
            if ctx["device"] == "cuda":
                torch.cuda.empty_cache()
    comparisons = []
    for seed in ctx["comparison"]["seeds"]:
        rows = {
            r["summary"]["arm"]: r
            for r in reports
            if r["summary"]["seed"] == seed and r["summary"]["readout"] == "Top-16"
        }
        a = rows["ggf"]
        aa = memory.required(s, "compact_" + a["tag"])
        for arm in ("continued", "SLiCS mask", "SLiCS image", "SVD mask", "SVD image"):
            b = rows[arm]
            ci = study.paired(aa, memory.required(s, "compact_" + b["tag"]), ctx["cfg"])
            va = a["summary"]
            vb = b["summary"]
            gap = va["capture"] - vb["capture"]
            apgap = va["ap"] - vb["ap"]
            cost = vb["reconstruction"]["ve"] - va["reconstruction"]["ve"]
            comparisons.append(
                dict(
                    seed=seed,
                    control=arm,
                    capture_gain=gap,
                    ap_gain=apgap,
                    ve_cost=cost,
                    paired_intervals=ci,
                    positive_within_cost=bool(
                        gap > 0
                        and apgap >= 0
                        and cost <= ctx["comparison"]["max_ve_cost"]
                    ),
                    control_checkpoint_eligible=vb["eligible"],
                )
            )
    out = dict(
        complete=True,
        models=[r["summary"] for r in reports],
        comparisons=comparisons,
        primary="GGF versus SLiCS mask, identical dictionary capacity and Top-16 readout.",
        interpretation="DINOv3 patch task adaptation, not published SLiCS retrieval replication. Reused holdout; no independent-test claim. Three starts share one dataset.",
        inference="No annotations at encoding; calibration selects evaluation group IDs only.",
        diagnostic_warnings=[
            dict(
                seed=r["summary"]["seed"],
                arm=r["summary"]["arm"],
                readout=r["summary"]["readout"],
                checkpoint_ineligible=r["summary"]["eligible"] is False,
                solver_nonconverged=r["summary"]
                .get("solver", {})
                .get("nonconverged", 0)
                if r["summary"].get("solver")
                else 0,
            )
            for r in reports
            if r["summary"]["eligible"] is False
            or (r["summary"].get("solver") or {}).get("nonconverged", 0) > 0
        ],
        decision="Report capture/AP/VE effects and paired intervals for all starts, including failures. A win over an ineligible or poorly solved comparator is inconclusive. No guaranteed win, convergence, optimality, SOTA or publication claim.",
    )
    s.save("comparison_summary", out)
    s.write("summary.json", out)
    tables_and_plots(ctx, reports, comparisons)
    memory.assert_immutable(ctx["stamps"])
    return out


def tables_and_plots(ctx, reports, comparisons):
    s = ctx["store"]
    flat = []
    for r in reports:
        v = r["summary"]
        flat.append(
            dict(
                seed=v["seed"],
                method=v["arm"],
                readout=v["readout"],
                capture_pct=100 * v["capture"],
                ap_pct=100 * v["ap"],
                ve_pct=100 * v["reconstruction"]["ve"],
                active_groups=v["reconstruction"]["active_groups"],
                checkpoint_eligible=v["eligible"],
                train_seconds=v["training_seconds"],
                audit_seconds=v["audit_seconds"],
                parameters=v["parameters"],
            )
        )
    with (s.results / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    with (s.results / "per_category.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "method",
                "readout",
                "category",
                "contrast_group",
                "detector_group",
                "capture",
                "ap",
                "objects",
                "positive_images",
            ],
        )
        w.writeheader()
        for r in reports:
            v = r["summary"]
            for c in r["categories"]:
                if c["category_index"] not in ctx["cfg"]["category_indices"]:
                    continue
                w.writerow(
                    dict(
                        seed=v["seed"],
                        method=v["arm"],
                        readout=v["readout"],
                        category=c["category"],
                        contrast_group=c["contrast"]["groups"][0],
                        detector_group=c["detection"]["group"],
                        capture=c["contrast"]["single_group_capture"],
                        ap=c["detection"]["ap"],
                        objects=c["contrast"]["object_count"],
                        positive_images=c["test_positive_images"],
                    )
                )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    arms = ["continued", "SVD image", "SLiCS image", "SVD mask", "SLiCS mask", "ggf"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, field, title in zip(
        axes,
        ["capture", "ap", "ve"],
        ["Object contrast capture", "Patch detection AP", "Reconstruction VE"],
    ):
        for j, arm in enumerate(arms):
            vals = [
                r[field + "_pct"]
                for r in flat
                if r["method"] == arm and r["readout"] == "Top-16"
            ]
            ax.bar(j, np.mean(vals), color="#008c82" if arm == "ggf" else "#9eaec8")
            ax.scatter(
                np.linspace(j - 0.1, j + 0.1, len(vals)),
                vals,
                c="black",
                s=16,
                zorder=3,
            )
        ax.set_xticks(range(len(arms)), arms, rotation=35, ha="right", fontsize=8)
        ax.set(title=title, ylabel="%")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "75 groups × 3 directions; Top-16; all three starts; reused 1,500-image holdout"
    )
    fig.tight_layout()
    folder = s.results / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"comparison.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def preflight(ctx):
    s = ctx["store"]
    prior = s.load("preflight")
    if prior is not None:
        return prior
    cfg = ctx["comparison"]
    x, masks, _ = batch(ctx, cfg["seeds"][0], 0)
    d = x.shape[-1]
    # Use training-supported initial dictionaries, not a misleading random-only timing.
    timings = []
    for kind in ("mask", "image"):
        D = init_slics(ctx, cfg["seeds"][0], kind)
        allow, known = support(
            masks, ctx["cfg"]["category_indices"], cfg, kind, ctx["device"]
        )
        take = torch.as_tensor(known, device=ctx["device"])
        xx = x.flatten(0, 1)[take]
        allow = allow[take]
        started = time.monotonic()
        z, solver = solve(xx, D, allow, cfg["nnls_iterations"], cfg["nnls_tolerance"])
        _, atom = atom_sweep(xx, D.clone(), z, allow, cfg["atom_nonincrease_guard"])
        study.sync(ctx["device"])
        train_sec = time.monotonic() - started
        m = SlicsModel(D, cfg)
        started = time.monotonic()
        encoded = m.encode(x.flatten(0, 1))
        study.sync(ctx["device"])
        infer = time.monotonic() - started
        timings.append(
            dict(
                kind=kind,
                train_seconds_per_batch=train_sec,
                inference_seconds_per_patch=infer / len(encoded),
                solver=solver,
                atom=atom,
                finite=bool(torch.isfinite(encoded).all()),
            )
        )
        del m, D
    total = cfg["warm_updates"] + cfg["guided_updates"]
    hours = (
        sum(r["train_seconds_per_batch"] for r in timings)
        * total
        * len(cfg["seeds"])
        / 3600
    )
    audit_hours = (
        sum(r["inference_seconds_per_patch"] for r in timings)
        * 196
        * (2000 + 400 * 11 + 1900 * 3)
        * len(cfg["seeds"])
        / 3600
    )
    out = dict(
        numerical_ok=all(r["finite"] for r in timings),
        timings=timings,
        rough_slics_training_hours=hours,
        rough_slics_primary_audit_hours=audit_hours,
        excluded="SGD, SVD initialization, validation variability, extra dense sensitivity, bootstraps, Drive and stalls. Single-batch extrapolations, not an ETA.",
        dictionary_atoms_each=225,
        groups_each=75,
        primary_max_active_groups=16,
        compute_matched=False,
        instruction="If solver warnings or costs are high, stop before training and send preflight.json. Do not silently change settings mid-run.",
    )
    s.save("preflight", out)
    s.write("preflight.json", out)
    return out


def export(ctx):
    return old.export(ctx)
