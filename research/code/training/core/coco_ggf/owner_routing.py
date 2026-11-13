"""One-seed, saved-GGF-start screen of category ownership and learned routing.

No baseline retraining, extraction, structural edits, or final-test access.
"""

import copy
import gc
import hashlib
import json
import time
import traceback
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from coco_coordinated import run as memory
from coco_ggf import study
from .owner_model import (
    OwnerGGF,
    MODES,
    distinct_owners,
    category_capture_matrix,
    calibrated_targets,
    supervised_terms,
)

EXPECTED = "95e7186b193ed3744ced01585a0ff0f4a198c7318a53ab23cde20b4adb5082fd"
EXPECTED_SOURCE = "eb291dc32c7989cd3f64b5aa4cf46c8369ab6c7fd50b764f7f2f8a326371f7df"
LABELS = dict(
    continued_ggf="Continued GGF",
    owner_topk="Distinct owners + native Top-K",
    owner_router="Distinct owners + learned routing",
)


def settings():
    return dict(
        version=1,
        study="ggf_distinct_owner_routing_screen",
        seeds=[21],
        updates=400,
        evaluate_every=100,
        local_save_every=20,
        drive_save_every=100,
        gradient_every=100,
        lr=3e-4,
        images_per_update=8,
        snr=0.1,
        lambda_capture=0.05,
        lambda_detection=0.01,
        grad_clip=5.0,
        assignment_images=512,
        assignment_min_images=3,
        assignment_pool_groups=256,
        max_ve_drop=0.001,
        max_ap_drop=0.001,
        max_relative_mse_growth=0.02,
        minimum_mean_capture_gain=0.001,
        bootstrap=200,
        training_seed=104071,
        decision_scope="Exploratory one-seed screen on previously inspected development images. No final test. Fresh seeds and untouched images required before a success claim.",
        no_structural_edits=True,
        owner_quota=1,
        active_groups=16,
        group_rank=3,
        total_groups=1024,
    )


def prepare(
    source_output,
    output,
    local,
    sha,
    versions,
    cfg=None,
    device="cuda",
    require_drive=True,
):
    cfg = copy.deepcopy(settings() if cfg is None else cfg)
    expected = settings()
    if cfg != expected:
        raise ValueError(
            "Use the frozen one-seed screening settings; new protocols need a separate notebook"
        )
    local = Path(local)
    source = memory.FrozenStore(
        Path(source_output) / "checkpoints" / EXPECTED[:20], local / "read_ggf"
    )
    if source.run_id != EXPECTED or source.config["source_sha"] != EXPECTED_SOURCE:
        raise ValueError("Wrong successful GGF parent; no retraining fallback")
    activation_output = (
        Path(source_output).parent / "AGF_COCO_stratified_confirmation_v1"
    )
    original = memory.FrozenStore(
        activation_output / "checkpoints" / source.config["parent_run_id"][:20],
        local / "read_activations",
    )
    if original.run_id != source.config["parent_run_id"]:
        raise ValueError("Wrong frozen activation parent")
    dest = Path(output).resolve()
    for root in (Path(source_output).resolve(), activation_output.resolve()):
        if dest == root or root in dest.parents or dest in root.parents:
            raise ValueError("Every input is read-only; use a separate output folder")
    frozen = memory.required(source, "all_models_frozen")
    if not frozen["complete"]:
        raise ValueError("Successful parent has not finished")
    base = copy.deepcopy(source.config["native_config"])
    inherited = copy.deepcopy(source.config["config"])
    if (base["groups"], base["block_rank"], base["active_groups"]) != (1024, 3, 16):
        raise ValueError("Source dictionary/budgets differ")
    store = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=sha,
            versions=versions,
            source_run=source.run_id,
            source_sha_parent=source.config["source_sha"],
            activation_run=original.run_id,
            source_numerical_versions=source.config["numerical_versions"],
            device=device,
        ),
        local / "rescue",
        require_drive,
    )
    memory.memory_safe_metrics()
    provenance = json.loads((original.results / "provenance.json").read_text())
    norm = memory.required(original, "normalization")
    if (
        not np.isfinite(norm["scale"])
        or norm["scale"] <= 0
        or not np.isfinite(norm["mean"]).all()
    ):
        raise ValueError("Saved training normalization is invalid")
    sets = [set(v) for v in provenance["splits"].values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Activation split overlap")
    packs = {}
    for split in ("train", "select"):
        store.event("loading_verified_activation_cache", split=split)
        packs[split] = memory.normalized_pack(
            original, split, provenance, norm, local, inherited["extract_chunk"]
        )
    if set(packs["train"]["ids"]) & set(packs["select"]["ids"]):
        raise ValueError("Training/development overlap")
    if len(packs["train"]["ids"]) != 2000 or len(packs["select"]["ids"]) != 400:
        raise ValueError(
            "Source training/development sizes differ from the declared successful recipe"
        )
    if any(tuple(p["x"].shape[1:]) != (196, 768) for p in packs.values()):
        raise ValueError(
            "Expected 196 final-layer DINOv3 768-dimensional patches per image"
        )
    cfg_eval = dict(
        inherited, device=device, audit_image_batch=2, preview_images=0, batch=1024
    )
    ctx = dict(
        store=store,
        source=source,
        original=original,
        cfg=cfg,
        eval_cfg=cfg_eval,
        base=base,
        packs=packs,
        local=local,
        device=device,
        frozen=frozen,
        provenance=provenance,
        stamps=memory.source_snapshot([source, original]),
    )
    store.write(
        "protocol_settings.json",
        dict(
            settings=cfg,
            annotation_role="Training-only owner assignment, head calibration and loss. No masks enter model inference.",
            starts="Saved successful ggf_s21 BEST dictionary, fresh common Adam; no native training.",
            comparison="Three arms, same full dictionary, samples, max updates, losses and hard active budget. Readout assignment and routing changes are separated.",
            gates="Independent learned category gates; over 16 predicted categories are truncated and counted. Residual bank fills remaining slots.",
            final_test_opened=False,
            numerical_version_policy="Reevaluate all arms in the current environment; do not reuse old metric values.",
        ),
    )
    return ctx


def source_start(ctx, seed):
    row = next(
        r for r in ctx["frozen"]["models"] if r["seed"] == seed and r["arm"] == "ggf"
    )
    st = memory.required(ctx["source"], row["key"])
    weights = st[row["field"]]
    if training.weight_sha(weights) != row["weights_sha"]:
        raise ValueError("Saved successful GGF weights changed")
    if st["best_update"] != st["updates"]:
        raise ValueError(
            "Need matching saved best head; do not pair final head with earlier dictionary"
        )
    targets = memory.required(ctx["source"], f"targets_s{seed}")
    return weights, st["head"], targets, row


def assignment_indices(pack, cohort, cfg, eval_cfg):
    """Training-mask coverage, including actual label-blind detection samples."""
    evidence = np.zeros((len(pack["ids"]), len(cohort)), bool)
    for i, (mask, iid) in enumerate(zip(pack["masks"], pack["ids"])):
        if any(c < 0 or c >= mask["fraction"].shape[1] for c in cohort):
            raise ValueError("Training category coverage/schema mismatch")
        pick = np.random.default_rng(eval_cfg["patch_sample_seed"] + int(iid)).choice(
            196, eval_cfg["detection_patches"], replace=False
        )
        known = mask["crowd"] <= eval_cfg["negative_fraction"]
        for j, c in enumerate(cohort):
            has_positive = (
                (
                    mask["fraction"][pick, c].astype(float) / 255
                    >= eval_cfg["positive_fraction"]
                )
                & known[pick]
            ).any()
            bg = (
                mask["fraction"][:, c].astype(float) / 255
                <= eval_cfg["negative_fraction"]
            ) & known
            good = any(
                o["category"] == c
                and (np.asarray(o["fraction"]) * known).sum()
                >= eval_cfg["min_object_patch_mass"]
                for o in mask["instances"]
            )
            evidence[i, j] = (
                has_positive and good and bg.sum() >= eval_cfg["min_background_patches"]
            )
    quota = cfg["assignment_min_images"]
    counts = np.zeros(len(cohort), int)
    selected = []
    free = np.ones(len(evidence), bool)
    rng = np.random.default_rng(cfg["training_seed"] + 9)
    tie = rng.random(len(evidence))
    while (counts < quota).any():
        need = np.maximum(quota - counts, 0)
        gain = evidence @ need
        gain[~free] = -1
        i = int(np.lexsort((tie, gain))[-1])
        if gain[i] <= 0 or len(selected) >= cfg["assignment_images"]:
            missing = [int(cohort[j]) for j in np.flatnonzero(counts < quota)]
            raise ValueError(
                "Training owner-assignment coverage insufficient: " + str(missing)
            )
        selected.append(i)
        free[i] = False
        counts += evidence[i]
    rest = rng.permutation(np.flatnonzero(free))[
        : max(0, cfg["assignment_images"] - len(selected))
    ]
    selected.extend(rest.tolist())
    return np.array(selected), dict(
        category_indices=cohort,
        usable_images=(evidence[selected].sum(0)).tolist(),
        minimum_images=quota,
        ids=[int(pack["ids"][i]) for i in selected],
        rule="Training coverage only. No development scores or labels choose owners.",
    )


@torch.no_grad()
def raw_owner_heat(model, pack, owners, cfg, device):
    heat = []
    for i, iid in enumerate(pack["ids"]):
        x = torch.as_tensor(np.array(pack["x"][i], copy=True), device=device)
        raw = (x @ model.W_enc + model.b_enc).reshape(
            196, model.n_groups, model.group_size
        )
        pick = np.random.default_rng(cfg["patch_sample_seed"] + int(iid)).choice(
            196, cfg["detection_patches"], replace=False
        )
        heat.append(raw[pick][:, owners].norm(dim=-1).cpu().numpy())
    return np.concatenate(heat)


def owner_plan(ctx, seed):
    s = ctx["store"]
    key = f"owners_s{seed}"
    old = s.load(key)
    if old is not None:
        s.event(
            "reused_owner_assignment", seed=seed, assignment_sha=old["assignment_sha"]
        )
        return old
    weights, head, targets, row = source_start(ctx, seed)
    cohort = [t["category_index"] for t in targets]
    if sorted(cohort) != sorted(ctx["eval_cfg"]["category_indices"]):
        raise ValueError("Training cohort differs; never drop difficult categories")
    indices, coverage = assignment_indices(
        ctx["packs"]["train"], cohort, ctx["cfg"], ctx["eval_cfg"]
    )
    train = ctx["packs"]["train"]
    pack = dict(
        x=np.array(train["x"][indices], copy=True),
        ids=[train["ids"][i] for i in indices],
        masks=[train["masks"][i] for i in indices],
    )
    model = core.new_model("vanilla", ctx["base"], weights, ctx["device"])
    s.event("owner_assignment_audit_start", seed=seed, images=len(indices))
    audit = metrics.audit_arrays(
        model, pack["x"], pack["masks"], pack["ids"], ctx["eval_cfg"], ctx["device"]
    )
    priority = sorted(
        {t[k] for t in targets for k in ("contrast_group", "detector_group")}
    )
    energy = np.square(audit["heat"]).sum(0)
    pool = (
        priority
        + [int(g) for g in np.argsort(-energy) if int(g) not in priority][
            : max(0, ctx["cfg"]["assignment_pool_groups"] - len(priority))
        ]
    )
    if len(pool) < len(cohort):
        raise ValueError("Owner candidate pool too small")
    sub = dict(
        audit,
        atoms=audit["atoms"][pool],
        dz=audit["dz"][:, pool],
        heat=audit["heat"][:, pool],
    )
    captures = category_capture_matrix(sub, cohort)
    y, valid = metrics.binary_labels(sub, ctx["eval_cfg"])
    ap = metrics.ap_matrix(sub["heat"], y[:, cohort], valid[:, cohort]).T
    local_owners, score = distinct_owners(captures, ap)
    owners = np.array(pool)[local_owners]
    names = {
        c["index"] if "index" in c else i: c["name"]
        for i, c in enumerate(ctx["provenance"]["categories"])
    }
    owner_targets, owner_head = calibrated_targets(
        sub, dict(cfg=ctx["eval_cfg"], names=names), cohort, local_owners, captures
    )
    for t, g in zip(owner_targets, owners):
        t["contrast_group"] = t["detector_group"] = int(g)
    raw_heat = raw_owner_heat(
        model, pack, owners.tolist(), ctx["eval_cfg"], ctx["device"]
    )
    raw_audit = dict(sub, heat=raw_heat)
    router_targets, router_head = calibrated_targets(
        raw_audit,
        dict(cfg=ctx["eval_cfg"], names=names),
        cohort,
        np.arange(len(cohort)),
        np.zeros((len(cohort), len(cohort))),
    )
    for t, g, ot in zip(router_targets, owners, owner_targets):
        t["contrast_group"] = t["detector_group"] = int(g)
        t["training_capture"] = ot["training_capture"]
    assignment_sha = identity(
        dict(
            cohort=cohort,
            owners=owners.tolist(),
            ids=coverage["ids"],
            weights=row["weights_sha"],
        )
    )
    out = dict(
        targets=owner_targets,
        head=owner_head,
        router_targets=router_targets,
        router_head=router_head,
        owners=owners.tolist(),
        cohort=cohort,
        coverage=coverage,
        pool=pool,
        assignment_sha=assignment_sha,
        source_weights_sha=row["weights_sha"],
        training_score=score.tolist(),
        training_capture_matrix=captures,
        training_ap_matrix=ap,
        legacy_collisions={
            field: len(targets) - len({t[field] for t in targets})
            for field in ("contrast_group", "detector_group")
        },
        selection_scope="Exact matching within a training-selected candidate pool, not a globally optimal semantic assignment.",
    )
    s.save(key, out)
    s.write(
        f"assignments/s{seed}.json",
        {
            k: v
            for k, v in out.items()
            if k
            not in (
                "head",
                "router_head",
                "training_capture_matrix",
                "training_ap_matrix",
            )
        },
    )
    folder = s.results / "assignments"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(
        folder / f"s{seed}_candidate_scores.npz",
        capture=captures,
        ap=ap,
        score=captures + 0.2 * ap,
        category_indices=np.array(cohort),
        group_ids=np.array(pool),
        owner_ids=owners,
    )
    s.write(f"assignments/s{seed}_original_readouts.json", targets)
    alternatives = []
    for j, c in enumerate(cohort):
        score_row = captures[j] + 0.2 * ap[j]
        best = np.argsort(-score_row)[:5]
        chosen = int(local_owners[j])
        alternatives.append(
            dict(
                category=names[c],
                category_index=int(c),
                owner=int(owners[j]),
                assigned_score=float(score_row[chosen]),
                unconstrained_best_group=int(pool[best[0]]),
                assignment_score_cost=float(score_row[best[0]] - score_row[chosen]),
                alternatives=[
                    dict(
                        group=int(pool[g]),
                        capture=float(captures[j, g]),
                        ap=float(ap[j, g]),
                        score=float(score_row[g]),
                    )
                    for g in best
                ],
            )
        )
    s.write(f"assignments/s{seed}_alternatives.json", alternatives)
    s.event(
        "owner_assignment_frozen",
        seed=seed,
        owners=len(owners),
        residual_groups=ctx["base"]["groups"] - len(owners),
        assignment_sha=assignment_sha,
    )
    del model, audit, sub, pack
    gc.collect()
    return out


def make(ctx, seed, arm, weights=None):
    initial, oldhead, oldtargets, row = source_start(ctx, seed)
    plan = owner_plan(ctx, seed)
    if arm == "continued_ggf":
        targets, head, rt = oldtargets, oldhead, None
    else:
        targets, rt = plan["targets"], plan["router_targets"]
        head = plan["router_head"] if arm == "owner_router" else plan["head"]
    m = OwnerGGF(
        initial,
        ctx["base"]["groups"],
        ctx["base"]["block_rank"],
        ctx["base"]["active_groups"],
        targets,
        head,
        arm,
        rt,
    ).to(ctx["device"])
    if weights is not None:
        m.load_state_dict(weights)
    return m, row["weights_sha"], plan["assignment_sha"]


def scalar_metrics(report):
    return dict(
        capture=report["macro_capture"],
        ap=report["macro_ap"],
        ve=report["reconstruction"]["ve"],
        mse=report["reconstruction"]["mse"],
    )


def quality_guard(report, reference, cfg):
    v, ref = scalar_metrics(report), scalar_metrics(reference)
    reasons = []
    if any(x is None or not np.isfinite(x) for x in v.values()):
        reasons.append("nonfinite_metrics")
    else:
        if v["ve"] < ref["ve"] - cfg["max_ve_drop"]:
            reasons.append("reconstruction_ve_drop")
        if v["mse"] > ref["mse"] * (1 + cfg["max_relative_mse_growth"]):
            reasons.append("reconstruction_mse_growth")
        if v["ap"] < ref["ap"] - cfg["max_ap_drop"]:
            reasons.append("detection_ap_drop")
    return dict(
        passed=not reasons,
        reasons=reasons,
        reference=ref,
        measured=v,
        note="Explicit exploratory quality margins, not statistically calibrated equivalence bounds.",
    )


@torch.no_grad()
def validate(ctx, model, reference=None, with_audit=False):
    pack = ctx["packs"]["select"]
    cfg = ctx["eval_cfg"]
    audit = metrics.audit_arrays(
        model, pack["x"], pack["masks"], pack["ids"], cfg, ctx["device"]
    )
    report = core.category_scores(audit, model.targets, cfg)
    report["reconstruction"] = training.measure(
        model, pack["x"].reshape(-1, model.d), ctx["device"], cfg["batch"]
    )
    report["routing"] = routing_counts(ctx, model)
    if len(report["categories"]) != len(model.targets) or any(
        r["capture"] is None or r["ap"] is None for r in report["categories"]
    ):
        raise ValueError(
            "Development evidence insufficient; do not change the category denominator"
        )
    return (report, audit) if with_audit else report


def common_reference(ctx, seed):
    key = f"common_reference_s{seed}"
    old = ctx["store"].load(key)
    if old is not None:
        return old
    m, _, _ = make(ctx, seed, "continued_ggf")
    v = validate(ctx, m)
    ctx["store"].save(key, v)
    del m
    gc.collect()
    return v


def preflight(ctx):
    s = ctx["store"]
    old = s.load("preflight")
    if old is not None:
        return old
    seed = ctx["cfg"]["seeds"][0]
    common_reference(ctx, seed)
    pack = ctx["packs"]["train"]
    cfg = ctx["cfg"]
    times = {}
    vals = {}
    for arm in MODES:
        m, _, _ = make(ctx, seed, arm)
        opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
        x = torch.as_tensor(
            np.array(pack["x"][: cfg["images_per_update"]], copy=True),
            device=ctx["device"],
        )
        masks = pack["masks"][: cfg["images_per_update"]]
        elapsed = []
        for i in range(4):
            study.sync(ctx["device"])
            begun = time.monotonic()
            m.train()
            native, _ = m.loss(x.reshape(-1, m.d))
            cap, det, _, _ = supervised_terms(m, x, masks, ctx["eval_cfg"])
            loss = native + cfg["lambda_capture"] * cap + cfg["lambda_detection"] * det
            if not torch.isfinite(loss):
                raise FloatingPointError("Invalid preflight objective")
            opt.zero_grad()
            loss.backward()
            opt.step()
            m.normalize_decoder()
            study.sync(ctx["device"])
            if i:
                elapsed.append(time.monotonic() - begun)
        times[arm] = float(np.median(elapsed))
        begun = time.monotonic()
        v = validate(ctx, m)
        vals[arm] = time.monotonic() - begun
        del m, opt, x
        gc.collect()
    n = len(ctx["cfg"]["seeds"])
    checks = 1 + cfg["updates"] // cfg["evaluate_every"]
    estimate = n * (cfg["updates"] * sum(times.values()) + checks * sum(vals.values()))
    out = dict(
        numerical_ok=True,
        seconds_per_update=times,
        full_development_seconds=vals,
        total_training_fits=3 * n,
        max_updates_per_fit=cfg["updates"],
        rough_remaining_minutes=[estimate / 60, 2 * estimate / 60],
        excludes="Checkpoint/Drive stalls, cache loading, final diagnostics, assignment and bootstrap. Scenarios, not an ETA.",
        same_dictionary_directions=3072,
        same_active_direction_cap=48,
        masks_used_by_inference=False,
        final_test_opened=False,
        preflight_updates_discarded=True,
    )
    s.save("preflight", out)
    s.write("preflight.json", out)
    return out


def train_arm(ctx, seed, arm, max_updates=None):
    s = ctx["store"]
    cfg = ctx["cfg"]
    key = f"s{seed}_{arm}"
    torch.manual_seed(cfg["training_seed"] + seed)
    if ctx["device"].startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["training_seed"] + seed)
    model, start_sha, assignment_sha = make(ctx, seed, arm)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    rng = np.random.default_rng(cfg["training_seed"] + seed)
    reference = common_reference(ctx, seed)
    st = s.load(key)
    if st is None:
        val = validate(ctx, model)
        guard = quality_guard(val, reference, cfg)
        st = dict(
            updates=0,
            target=cfg["updates"],
            start_sha=start_sha,
            assignment_sha=assignment_sha,
            arm=arm,
            weights=training.weights(model),
            best=None,
            best_update=None,
            best_capture=None,
            history=[dict(update=0, metrics=val, quality_guard=guard)],
            seconds=0.0,
            batch_digest="",
        )
        if guard["passed"]:
            st.update(
                best=training.weights(model),
                best_update=0,
                best_capture=val["macro_capture"],
            )
        s.write("histories/" + key + ".json", st["history"])
        s.event(
            "initial_arm_metrics",
            seed=seed,
            arm=arm,
            updates=0,
            **scalar_metrics(val),
            quality_reasons=guard["reasons"],
        )
    else:
        if (st["target"], st["start_sha"], st["assignment_sha"], st["arm"]) != (
            cfg["updates"],
            start_sha,
            assignment_sha,
            arm,
        ):
            raise ValueError("Resume identity changed")
        model.load_state_dict(st["weights"])
        opt.load_state_dict(st["optimizer"])
        rng.bit_generator.state = st["sample_rng"]
        torch.set_rng_state(st["torch_rng"])
        if ctx["device"].startswith("cuda") and st["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
    print(key, "resume", st["updates"], "/", cfg["updates"], flush=True)
    stop = cfg["updates"] if max_updates is None else min(cfg["updates"], max_updates)

    def save(drive=True):
        st.update(
            weights=training.weights(model),
            optimizer=copy.deepcopy(opt.state_dict()),
            sample_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if ctx["device"].startswith("cuda")
            else None,
        )
        s.save(key, st, drive)

    try:
        while st["updates"] < stop:
            ix = rng.integers(
                len(ctx["packs"]["train"]["ids"]), size=cfg["images_per_update"]
            )
            st["batch_digest"] = hashlib.sha256(
                st["batch_digest"].encode() + ix.tobytes()
            ).hexdigest()
            pack = ctx["packs"]["train"]
            x = torch.as_tensor(
                np.array(pack["x"][ix], copy=True), device=ctx["device"]
            )
            model.train()
            begun = time.monotonic()
            xx = x.reshape(-1, model.d)
            noisy = (
                xx + cfg["snr"] * xx.std() * torch.randn_like(xx) if cfg["snr"] else xx
            )
            native, _ = model.loss(noisy, target=xx)
            cap, det, objects, coverage = supervised_terms(
                model, x, [pack["masks"][i] for i in ix], ctx["eval_cfg"], details=True
            )
            terms = dict(
                native=native,
                capture=cfg["lambda_capture"] * cap,
                detection=cfg["lambda_detection"] * det,
            )
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective")
            u = st["updates"] + 1
            diagnostics = (
                study.component_gradients(model, terms)
                if u % cfg["gradient_every"] == 0
                else None
            )
            opt.zero_grad()
            loss.backward()
            grad = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            )
            eg = float(model.W_enc.grad.norm())
            dg = float(model.W_dec.grad.norm())
            hg = float(model.head.grad.norm())
            if not np.isfinite([grad, eg, dg, hg]).all():
                raise FloatingPointError("Nonfinite training gradients")
            opt.step()
            model.normalize_decoder()
            study.sync(ctx["device"])
            st["updates"] = u
            st["seconds"] += time.monotonic() - begun
            if u % 20 == 0:
                print(key, u, "/", cfg["updates"], flush=True)
            if u % cfg["evaluate_every"] == 0 or u == cfg["updates"]:
                val = validate(ctx, model)
                guard = quality_guard(val, reference, cfg)
                if guard["passed"] and (
                    st["best_capture"] is None
                    or val["macro_capture"] > st["best_capture"]
                ):
                    st.update(
                        best=training.weights(model),
                        best_update=u,
                        best_capture=val["macro_capture"],
                    )
                row = dict(
                    update=u,
                    metrics=val,
                    quality_guard=guard,
                    losses={k: float(v.detach()) for k, v in terms.items()},
                    raw_capture_loss=float(cap.detach()),
                    raw_detection_loss=float(det.detach()),
                    encoder_gradient_norm=eg,
                    decoder_gradient_norm=dg,
                    head_gradient_norm=hg,
                    gradient_norm=grad,
                    component_gradients=diagnostics,
                    training_batch_coverage=coverage,
                    objects_in_batch=objects,
                    seconds_per_update=st["seconds"] / u,
                    best_update=st["best_update"],
                )
                st["history"].append(row)
                s.write("histories/" + key + ".json", st["history"])
                s.event(
                    "training_checkpoint",
                    seed=seed,
                    arm=arm,
                    updates=u,
                    target=cfg["updates"],
                    **scalar_metrics(val),
                    quality_reasons=guard["reasons"],
                    best_update=st["best_update"],
                )
            if u % 20 == 0:
                s.event(
                    "training_step",
                    seed=seed,
                    arm=arm,
                    updates=u,
                    target=cfg["updates"],
                    losses={k: float(v.detach()) for k, v in terms.items()},
                    encoder_gradient_norm=eg,
                    decoder_gradient_norm=dg,
                    head_gradient_norm=hg,
                    gradient_norm_before_clip=grad,
                    objects_in_batch=objects,
                    batch_digest=st["batch_digest"],
                    seconds_per_update=st["seconds"] / u,
                )
            if u % cfg["drive_save_every"] == 0 or u == cfg["updates"]:
                save()
            elif u % cfg["local_save_every"] == 0:
                save(False)
        save()
    except BaseException as exc:
        save(False)
        try:
            save()
            s.event(
                "pipeline_error",
                stage_name="train_arm",
                arm=arm,
                updates=st["updates"],
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        except BaseException:
            pass
        raise
    del model, opt
    gc.collect()
    if ctx["device"].startswith("cuda"):
        torch.cuda.empty_cache()
    return st


def train(ctx, max_updates=None):
    rows = []
    for seed in ctx["cfg"]["seeds"]:
        arms = []
        for arm in MODES:
            state = train_arm(ctx, seed, arm, max_updates)
            arms.append(state)
            rows.append(
                dict(
                    seed=seed,
                    arm=arm,
                    updates=state["updates"],
                    best_update=state["best_update"],
                    best_capture=state["best_capture"],
                    eligible=state["best"] is not None,
                )
            )
        if len({a["batch_digest"] for a in arms}) != 1:
            raise RuntimeError("Training minibatches differ across arms")
        if len({a["start_sha"] for a in arms}) != 1:
            raise RuntimeError("Dictionary starts differ across arms")
    complete = all(r["updates"] == ctx["cfg"]["updates"] for r in rows)
    result = dict(complete=complete, models=rows, final_test_opened=False)
    ctx["store"].save("models_frozen", result)
    ctx["store"].write("models_frozen.json", result)
    memory.assert_immutable(ctx["stamps"])
    return result


@torch.no_grad()
def routing_counts(ctx, model):
    """Actual hard routes, including per-category missed positive patches."""
    pack = ctx["packs"]["select"]
    cfg = ctx["eval_cfg"]
    owners = model.owner_ids.tolist()
    cohort = [t["category_index"] for t in model.targets]
    n = len(cohort)
    names = [
        "positive",
        "negative",
        "requested_positive",
        "requested_negative",
        "admitted_positive",
        "admitted_negative",
        "truncated_positive",
        "active_positive",
        "active_negative",
    ]
    counts = {name: np.zeros(n, np.int64) for name in names}
    route = dict(
        patches=0,
        requested=0.0,
        admitted=0.0,
        truncated=0.0,
        overflow_patches=0,
        residual=0.0,
        active=0.0,
        max_active_groups=0,
        total_group_energy=0.0,
        owner_group_energy=0.0,
    )
    for lo in range(0, len(pack["x"]), 2):
        x = torch.as_tensor(
            np.array(pack["x"][lo : lo + 2], copy=True).reshape(-1, model.d),
            device=ctx["device"],
        )
        if model.mode == "owner_router":
            z, logits, r = model.routed(x, True)
            requested = (logits >= 0).cpu().numpy()
            admitted = r["owner_admitted"].bool().cpu().numpy()
            route["requested"] += float(r["requested_owner_count"].sum())
            route["admitted"] += float(r["admitted_owner_count"].sum())
            route["truncated"] += float(r["truncated_owner_count"].sum())
            route["overflow_patches"] += int((r["truncated_owner_count"] > 0).sum())
            route["residual"] += float(r["residual_count"].sum())
        else:
            z = model.encode(x)
            requested = admitted = None
        on = z.norm(dim=-1) > 1e-6
        own_on = on[:, owners].cpu().numpy()
        route["active"] += float(on.sum())
        route["patches"] += len(x)
        route["max_active_groups"] = max(
            route["max_active_groups"], int(on.sum(1).max())
        )
        route["total_group_energy"] += float(z.square().sum())
        route["owner_group_energy"] += float(z[:, sorted(set(owners))].square().sum())
        fractions = (
            np.concatenate(
                [mask["fraction"][:, cohort] for mask in pack["masks"][lo : lo + 2]]
            ).astype(float)
            / 255
        )
        known = (
            np.concatenate([mask["crowd"] for mask in pack["masks"][lo : lo + 2]])
            <= cfg["negative_fraction"]
        )
        positive = (fractions >= cfg["positive_fraction"]) & known[:, None]
        negative = (fractions <= cfg["negative_fraction"]) & known[:, None]
        counts["positive"] += positive.sum(0)
        counts["negative"] += negative.sum(0)
        counts["active_positive"] += (own_on & positive).sum(0)
        counts["active_negative"] += (own_on & negative).sum(0)
        if requested is not None:
            for label, mask in [("positive", positive), ("negative", negative)]:
                counts["requested_" + label] += (requested & mask).sum(0)
                counts["admitted_" + label] += (admitted & mask).sum(0)
            counts["truncated_positive"] += (requested & ~admitted & positive).sum(0)
    if route["max_active_groups"] > ctx["base"]["active_groups"]:
        raise RuntimeError("Inference exceeded matched active budget")
    route["category_counts"] = [
        dict(
            category=t["category"],
            category_index=t["category_index"],
            group=int(owners[j]),
            **{name: int(value[j]) for name, value in counts.items()},
        )
        for j, t in enumerate(model.targets)
    ]
    route["request_counts_applicable"] = model.mode == "owner_router"
    return route


@torch.no_grad()
def diagnose(ctx, model, seed, audit=None, route=None):
    pack = ctx["packs"]["select"]
    cfg = ctx["eval_cfg"]
    plan = owner_plan(ctx, seed)
    if audit is None:
        audit = metrics.audit_arrays(
            model, pack["x"], pack["masks"], pack["ids"], cfg, ctx["device"]
        )
    y, valid = metrics.binary_labels(audit, cfg)
    cohort = plan["cohort"]
    owners = plan["owners"]
    matrix = metrics.ap_matrix(audit["heat"][:, owners], y[:, cohort], valid[:, cohort])
    category = []
    cats = np.array([o["category"] for o in audit["objects"]])
    D = audit["atoms"]
    owner_set = set(owners)
    for j, c in enumerate(cohort):
        ids = np.flatnonzero(cats == c)
        t = model.targets[j]
        target = audit["dx"][ids]
        dz = audit["dz"][ids]
        all_contrast = np.einsum("mgr,grd->md", dz, D, optimize=True)
        named = np.einsum("mgr,grd->md", dz[:, owners], D[owners], optimize=True)
        own = dz[:, t["contrast_group"]] @ D[t["contrast_group"]]
        residual = all_contrast - named
        strongest_other = int(
            np.nanargmax(np.where(np.arange(len(cohort)) == j, -np.inf, matrix[j]))
        )
        category.append(
            dict(
                category=t["category"],
                category_index=int(c),
                assigned_owner=int(owners[j]),
                actual_capture_group=t["contrast_group"],
                actual_detector_group=t["detector_group"],
                owner_capture=metrics.capture(target, own),
                all_group_capture=metrics.capture(target, all_contrast),
                assigned_owner_capture=metrics.capture(
                    target, dz[:, owners[j]] @ D[owners[j]]
                ),
                residual_bank_capture=metrics.capture(target, residual),
                residual_contrast_energy_ratio=float(
                    np.square(residual.astype(float)).sum()
                    / max(np.square(target.astype(float)).sum(), 1e-12)
                ),
                owner_group_active_on_positive_patches=float(
                    (audit["heat"][y[:, c] & valid[:, c], owners[j]] > 1e-6).mean()
                ),
                owner_ap_on_own_category=float(matrix[j, j]),
                strongest_other_category=ctx["provenance"]["categories"][
                    cohort[strongest_other]
                ]["name"],
                strongest_other_category_ap=float(matrix[j, strongest_other]),
                objects=len(ids),
            )
        )
    route = routing_counts(ctx, model) if route is None else route
    return dict(
        categories=category,
        owner_category_ap_matrix=matrix,
        category_indices=cohort,
        owner_group_ids=owners,
        routing=route,
        group_parameters=sum(p.numel() for p in model.parameters()),
        router_parameters=model.head.numel(),
        deployed_router_parameters=model.head.numel()
        if model.mode == "owner_router"
        else 0,
        quota="One distinct owned group per known category; >16 simultaneous predicted owners are truncated, not all guaranteed active.",
        interpretation="Residual capture/energy and cross-category AP are leakage/mixing diagnostics. They are not an exact feature-splitting rate or exclusive semantic recovery.",
    )


def paired_capture_interval(ctx, a, b, seed, audits=None):
    """Whole-image paired bootstrap of the fixed category denominator."""
    pack = ctx["packs"]["select"]
    cfg = ctx["eval_cfg"]
    prepared = []
    for j, m in enumerate((a, b)):
        audit = (
            audits[j]
            if audits is not None
            else metrics.audit_arrays(
                m, pack["x"], pack["masks"], pack["ids"], cfg, ctx["device"]
            )
        )
        groups = {t["category_index"]: t["contrast_group"] for t in m.targets}
        values = []
        for i, o in enumerate(audit["objects"]):
            c = o["category"]
            if c not in groups:
                continue
            dx = audit["dx"][i].astype(float)
            g = groups[c]
            pred = audit["dz"][i, g] @ audit["atoms"][g]
            values.append(
                (o["image_index"], c, float(dx @ dx), float(np.square(dx - pred).sum()))
            )
        prepared.append(values)
    cohort = [t["category_index"] for t in a.targets]
    cohort_map = {c: j for j, c in enumerate(cohort)}
    arrays = []
    for values in prepared:
        den = np.zeros((len(pack["ids"]), len(cohort)))
        err = den.copy()
        for i, c, d, e in values:
            den[i, cohort_map[c]] += d
            err[i, cohort_map[c]] += e
        arrays.append((den, err))
    if not np.array_equal(arrays[0][0], arrays[1][0]):
        raise RuntimeError("Paired object targets differ")
    rng = np.random.default_rng(ctx["cfg"]["training_seed"] + seed)
    gains = []
    invalid = 0
    for _ in range(ctx["cfg"]["bootstrap"]):
        ix = rng.integers(len(pack["ids"]), size=len(pack["ids"]))
        den = arrays[0][0][ix].sum(0)
        if (den <= 1e-12).any():
            invalid += 1
            continue
        gains.append(
            float(((arrays[0][1][ix].sum(0) - arrays[1][1][ix].sum(0)) / den).mean())
        )
    return dict(
        capture_gain_ci95=None
        if not gains
        else np.quantile(gains, [0.025, 0.975]).tolist(),
        valid_draws=len(gains),
        invalid_draws=invalid,
        requested_draws=ctx["cfg"]["bootstrap"],
        scope="Exploratory reused-development paired interval; no multiplicity correction or confirmation claim.",
    )


def summarize(ctx):
    s = ctx["store"]
    frozen = s.load("models_frozen")
    if frozen is None or not frozen["complete"]:
        raise RuntimeError("Finish all arms before comparison")
    rows = []
    comparisons = []
    for seed in ctx["cfg"]["seeds"]:
        reports = {}
        models = {}
        audits = {}
        for arm in MODES:
            st = s.load(f"s{seed}_{arm}")
            field = "best" if st["best"] is not None else "weights"
            model, _, _ = make(ctx, seed, arm, st[field])
            model.eval()
            report, audit = validate(ctx, model, with_audit=True)
            diag = diagnose(ctx, model, seed, audit, report["routing"])
            s.write(f"diagnostics/s{seed}_{arm}.json", diag)
            s.write(f"categories/s{seed}_{arm}.json", report["categories"])
            reports[arm] = report
            models[arm] = model
            audits[arm] = audit
            rows.append(
                dict(
                    seed=seed,
                    arm=arm,
                    label=LABELS[arm],
                    **scalar_metrics(report),
                    quality_eligible=st["best"] is not None,
                    selected_update=st["best_update"],
                    updates=st["updates"],
                    train_seconds=st["seconds"],
                    selected_weights_sha=training.weight_sha(st[field]),
                    dictionary_start_sha=st["start_sha"],
                    parameter_count=diag["group_parameters"],
                    final_checkpoint_metrics=scalar_metrics(
                        st["history"][-1]["metrics"]
                    ),
                    deployed_router_parameters=diag["deployed_router_parameters"],
                    active_groups=report["reconstruction"]["active_groups"],
                    max_active_groups=diag["routing"]["max_active_groups"],
                )
            )
        for arm, against in [
            ("owner_topk", "continued_ggf"),
            ("owner_router", "continued_ggf"),
            ("owner_router", "owner_topk"),
        ]:
            gain = reports[arm]["macro_capture"] - reports[against]["macro_capture"]
            guard = quality_guard(reports[arm], reports[against], ctx["cfg"])
            interval = paired_capture_interval(
                ctx, models[against], models[arm], seed, (audits[against], audits[arm])
            )
            comparisons.append(
                dict(
                    seed=seed,
                    arm=arm,
                    against=against,
                    capture_gain=gain,
                    ap_gain=reports[arm]["macro_ap"] - reports[against]["macro_ap"],
                    ve_gain=reports[arm]["reconstruction"]["ve"]
                    - reports[against]["reconstruction"]["ve"],
                    quality_guard=guard,
                    **interval,
                    screen_positive=bool(
                        guard["passed"]
                        and gain >= ctx["cfg"]["minimum_mean_capture_gain"]
                        and next(
                            r["quality_eligible"]
                            for r in rows
                            if r["seed"] == seed and r["arm"] == arm
                        )
                    ),
                )
            )
        del models, audits, audit
        gc.collect()
    memory.assert_immutable(ctx["stamps"])
    summary = dict(
        complete=True,
        models=rows,
        comparisons=comparisons,
        final_test_opened=False,
        scope=ctx["cfg"]["decision_scope"],
        minimum_screen_capture_gain=ctx["cfg"]["minimum_mean_capture_gain"],
        next_decisions="Owner-only wins: routing may be unnecessary. Router beats both: confirm fresh seeds/images. Both fail: inspect initial migration, dead owners, residual usage, category mixing and gradient conflicts; do not claim capacity exhaustion.",
        architecture="GGF with fixed distinct category owners; no block reorganization or dimension adaptation, so not an AGF structural win.",
    )
    s.save("summary", summary)
    s.write("summary.json", summary)
    s.write(
        "checkpoint_index.json",
        dict(
            root=str(s.root),
            models=[
                dict(
                    key=f"s{r['seed']}_{r['arm']}",
                    selected_field="best" if r["quality_eligible"] else "weights",
                    selected_update=r["selected_update"],
                    selected_weights_sha=r["selected_weights_sha"],
                    updates=r["updates"],
                )
                for r in rows
            ],
            note="Full model, optimizer, router buffers and RNGs remain in these verified Drive checkpoints. Results ZIP excludes large tensors.",
        ),
    )
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(figsize=(12, 3.5))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)
    for ax, metric, title in zip(
        axes,
        ["capture", "ap", "ve"],
        ["Object contrast capture", "Detection AP", "Reconstruction VE"],
    ):
        ax.bar(
            [LABELS[r["arm"]] for r in rows],
            [100 * r[metric] for r in rows],
            color=["#607d8b", "#8064a2", "#008e86"],
        )
        ax.set_title(title)
        ax.set_ylabel("%")
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    folder = s.results / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "owner_routing_comparison.png", dpi=160)
    return summary


def export(ctx):
    s = ctx["store"]
    target = (
        Path("/content") if s.require_drive and Path("/content").exists() else s.output
    )
    inventory = []
    for p in sorted(s.results.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest.json":
            inventory.append(
                dict(
                    path=p.relative_to(s.results).as_posix(),
                    bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                )
            )
    s.write(
        "artifact_manifest.json",
        dict(
            files=inventory,
            run_id=s.run_id,
            checkpoint_root=str(s.root),
            final_test_opened=False,
        ),
    )
    path = target / f"ggf_owner_routing_{s.run_id[:12]}_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob("*")):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(s.results).as_posix())
    return path
