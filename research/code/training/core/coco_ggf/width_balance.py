"""Five-arm mask-GGF group-width and bounded loss-balancing study.

All backbones, activation caches, the width-three control, and its warm-up
are reused read-only. This diagnostic does not perform AGF exchanges.
"""

import copy
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import data, training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import matched_baselines as matched, study
from .width_balance_fit import fit, Balancer, encoder_diagnostics
from .width_balance_warm import fit_warm

ARMS = {
    "mask3_fixed": (3, False, True, "Mask GGF: width 3, fixed losses"),
    "mask4_fixed": (4, False, True, "Mask GGF: width 4, fixed losses"),
    "mask3_balanced": (3, True, True, "Mask GGF: width 3, balanced losses"),
    "mask4_balanced": (4, True, True, "Mask GGF: width 4, balanced losses"),
    "vanilla4": (4, False, False, "Vanilla BSF: width 4"),
}
SEEDS = [21, 22, 23]


def settings():
    return dict(
        version=1,
        study="ggf_mask_width_balance",
        seeds=SEEDS,
        groups=1024,
        active_groups=16,
        phases=[2500, 500],
        phase_seeds=[71000, 82000],
        evaluate_every=250,
        save_every=50,
        balance=dict(
            kind="bounded_reconstruction_anchored_GradNorm_inspired",
            alpha=1.5,
            weight_lr=0.01,
            every=10,
            reference_batches=20,
            bounds=dict(capture=[0.0125, 0.2], detection=[0.0025, 0.2]),
        ),
        endpoint_rule="All final 3000-update endpoints; ineligible results explicitly retained",
        test_previously_inspected=True,
        new_structural_exchanges=0,
    )


def json_safe(value):
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def prepare(paths, output, local, source_sha, require_drive=True):
    # Includes the corrected identity-based discovery for multiple seed manifests.
    ctx = matched.prepare(
        paths, Path(output) / "input_verification", local, source_sha, require_drive
    )
    ctx["input_store"] = ctx["store"]
    ctx["recipe"] = settings()
    ctx["cfg"].update(evaluate_every=250, local_save_every=50)
    payload = dict(
        recipe=ctx["recipe"],
        source_sha=source_sha,
        source_runs={k: v.run_id for k, v in ctx["sources"].items()},
        versions=matched.versions(),
        base=ctx["base"],
        fit_config=ctx["cfg"],
        reused=ctx["reused"],
        test_identity=ctx["test_stamp"],
        note="Previously inspected cached test; model endpoints fixed before test loading; no AGF edits",
    )
    ctx["store"] = inputs.JournalStore(
        output, payload, Path(local) / "width_balance_rescue", require_drive
    )
    s = ctx["store"]
    # Fail before a long run if the original initializer or branch cache is missing.
    for seed in SEEDS:
        init = memory.required(ctx["sources"]["large"], f"large_mask_initial_s{seed}")
        D = init["D"]
        if D.shape != (225, 768) or not torch.isfinite(D).all():
            raise ValueError("Missing valid saved width-three mask dictionary")
        if (
            init["weights_sha"] != training.weight_sha(init["weights"])
            or [r["category_index"] for r in init["svd_report"]]
            != ctx["cfg"]["category_indices"]
        ):
            raise ValueError(
                "Saved mask initializer identity or category order changed"
            )
        ctx["checkpoint_stamps"].append(
            matched.source_key_stamp(
                ctx["sources"]["large"], f"large_mask_initial_s{seed}"
            )
        )
        if seed == 21:
            first = D.clone()
        elif not torch.equal(first, D):
            raise ValueError("Saved mask directions differ across seeds")
        warm = memory.required(ctx["sources"]["large"], f"large_mask_warm_s{seed}")
        if warm["initial_sha"] != training.weight_sha(init["weights"]):
            raise ValueError("Saved mask initializer/warm-up lineage differs")
        s.write(f"histories/reused_mask3_warm_s{seed}.json", warm["history"])
        for origin, key in [
            ("large", f"large_mask_ggf_s{seed}"),
            ("recovery", f"recovery_s{seed}_control"),
        ]:
            st = memory.required(ctx["sources"][origin], key)
            s.write("histories/reused_" + key + ".json", st["history"])
        del init, warm, st
    s.write("protocol.json", payload)
    s.write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    s.write(
        "split_roles.json",
        dict(
            train=ctx["provenance"]["splits"]["train"],
            development=ctx["provenance"]["splits"]["select"],
            test=ctx["test_ids"],
            test_previously_inspected=True,
        ),
    )
    matched.assert_sources(ctx)
    return ctx


def orthogonal_fourth(gram, saved):
    """Recover the fourth TRAIN eigen-direction without rotating saved axes."""
    eig, V = np.linalg.eigh(gram)
    leading = V[:, -3:].T[::-1]
    alignment = np.linalg.svd(saved @ leading.T, compute_uv=False)
    if alignment.min() < 0.999:
        raise ValueError(
            "Recomputed TRAIN eigenspace disagrees with saved first three directions"
        )
    direction = V[:, -4].copy()
    direction -= saved.T @ (saved @ direction)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Fourth direction is degenerate")
    return direction / norm, eig, alignment


def fourth_directions(ctx):
    store = ctx["store"]
    old = store.load("mask_dictionary4")
    if old is not None:
        saved = memory.required(ctx["sources"]["large"], "large_mask_initial_s21")[
            "D"
        ].reshape(75, 3, 768)
        if (
            old["sha"] != training.weight_sha({"D": old["D"]})
            or old["D"].shape != (300, 768)
            or not torch.equal(old["D"].reshape(75, 4, 768)[:, :3], saved)
        ):
            raise ValueError("Fourth-direction dictionary hash changed")
        return old["D"]
    # Preserve the first three directions byte-for-byte. Only the fourth needs
    # calculation from TRAIN features. The retained eigenspace must agree.
    D3 = (
        memory.required(ctx["sources"]["large"], "large_mask_initial_s21")["D"]
        .double()
        .numpy()
        .reshape(75, 3, 768)
    )
    pack = ctx["packs"]["train"]
    blocks = []
    records = []
    for slot, category in enumerate(ctx["cfg"]["category_indices"]):
        key = f"fourth_category_{category}"
        cached = store.load(key)
        if cached is None:
            started = time.monotonic()
            gram = np.zeros((768, 768), np.float64)
            count = 0
            for i, mask in enumerate(pack["masks"]):
                take = (mask["crowd"] <= 0.1) & (
                    mask["fraction"][:, category].astype(float) / 255 >= 0.5
                )
                X = np.asarray(pack["x"][i][take], np.float64)
                if len(X):
                    gram += X.T @ X
                    count += len(X)
            if count < 4:
                raise ValueError(
                    "Too few foreground patches for fourth direction: " + str(category)
                )
            direction, eig, alignment = orthogonal_fourth(gram, D3[slot])
            positive = negative = 0.0
            for i, mask in enumerate(pack["masks"]):
                take = (mask["crowd"] <= 0.1) & (
                    mask["fraction"][:, category].astype(float) / 255 >= 0.5
                )
                projection = np.asarray(pack["x"][i][take], np.float64) @ direction
                positive += np.square(np.maximum(projection, 0)).sum()
                negative += np.square(np.minimum(projection, 0)).sum()
            if negative > positive:
                direction *= -1
            block = torch.cat(
                [
                    torch.from_numpy(D3[slot]).float(),
                    torch.from_numpy(direction[None]).float(),
                ]
            )
            cached = dict(
                block=block,
                category=category,
                count=count,
                eigenvalues=eig[-4:][::-1].tolist(),
                eigenspace_alignment=alignment.tolist(),
                seconds=time.monotonic() - started,
            )
            store.save(key, cached)
        block = cached["block"]
        if (
            block.shape != (4, 768)
            or not torch.isfinite(block).all()
            or not torch.allclose(
                block[:3], torch.from_numpy(D3[slot]).float(), atol=0, rtol=0
            )
        ):
            raise ValueError("Invalid cached category initialization")
        if not torch.allclose(block @ block.T, torch.eye(4), atol=1e-4):
            raise ValueError("Mask initialization is not orthonormal")
        blocks.append(block)
        records.append({k: v for k, v in cached.items() if k != "block"})
        store.event(
            "fourth_direction_ready",
            category=slot + 1,
            total=75,
            training_patches=cached["count"],
        )
    D = torch.cat(blocks)
    store.save(
        "mask_dictionary4",
        dict(D=D, sha=training.weight_sha({"D": D}), records=records),
    )
    store.write("initialization/fourth_directions.json", records)
    return D


def initialize(base, D, seed):
    torch.manual_seed(seed)
    random = training.weights(training.make_model("vanilla", base, D.shape[1]))
    weights = copy.deepcopy(random)
    weights["W_dec"][: len(D)] = D
    weights["W_enc"] = weights["W_dec"].T.contiguous()
    weights["b_enc"].zero_()
    return weights, random


def paired_warm_rng(ctx, seed):
    state = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if ctx["device"].startswith("cuda") else None
    try:
        torch.manual_seed(seed)
        training.make_model(
            "vanilla", ctx["base"], ctx["packs"]["train"]["x"].shape[-1]
        )
        return torch.get_rng_state().clone()
    finally:
        torch.set_rng_state(state)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def subcontext(ctx, rank):
    sub = dict(ctx)
    sub["base"] = dict(ctx["base"], block_rank=rank, local_save_updates=50)
    return sub


def get_weights(ctx, row):
    store = ctx["sources"]["recovery"] if row["origin"] == "reused" else ctx["store"]
    state = memory.required(store, row["key"])
    w = state["weights"]
    if training.weight_sha(w) != row["weights_sha"]:
        raise ValueError("Frozen weights changed: " + row["arm"])
    if (
        row.get("head_sha") is not None
        and training.weight_sha({"head": state["head"]}) != row["head_sha"]
    ):
        raise ValueError("Frozen detection head changed: " + row["arm"])
    return w


def verify_catalog(ctx, frozen):
    expected = {(seed, arm) for seed in SEEDS for arm in ARMS}
    if (
        not frozen.get("complete")
        or len(frozen["models"]) != 15
        or {(r["seed"], r["arm"]) for r in frozen["models"]} != expected
    ):
        raise ValueError("All fifteen endpoints must freeze before test access")
    for row in frozen["models"]:
        rank = ARMS[row["arm"]][0]
        if row["rank"] != rank or row["targets_sha"] != identity(row["targets"]):
            raise ValueError("Frozen configuration changed")
        matched.check_targets(row["targets"], ctx["cfg"])
        w = get_weights(ctx, row)
        matched.valid_weights(w, "vanilla", subcontext(ctx, rank)["base"])
        del w


def freeze(ctx):
    s = ctx["store"]
    old = s.load("all_models_frozen")
    if old is not None:
        verify_catalog(ctx, old)
        return old
    D = fourth_directions(ctx)
    x = ctx["packs"]["train"]["x"].reshape(-1, 768)
    val = ctx["packs"]["select"]["x"].reshape(-1, 768)
    rows = []
    for seed in SEEDS:
        ref = next(
            r["native_reference_mse"] for r in ctx["reused"] if r["seed"] == seed
        )
        warm3 = memory.required(ctx["sources"]["large"], f"large_mask_warm_s{seed}")
        targets3 = memory.required(
            ctx["sources"]["large"], f"large_mask_targets_s{seed}"
        )
        sub = subcontext(ctx, 4)
        mask_initial, random_initial = initialize(sub["base"], D, seed)
        warm4 = {}
        targets4 = {}
        for kind, init in [("mask", mask_initial), ("vanilla", random_initial)]:
            key = f"warm4_{kind}_s{seed}"
            warm = fit_warm(
                s,
                key,
                "vanilla",
                x,
                val,
                sub["base"],
                seed,
                sub["base"]["lrs"][0],
                ctx["device"],
                initial=init,
                drive_every=50,
                pair_rng=paired_warm_rng(ctx, seed),
            )
            if warm["updates"] != 7640 or warm["best"] is None:
                raise ValueError("Incomplete width-four warm-up")
            if warm["sampled_batches_sha"] != warm3["sampled_batches_sha"]:
                raise ValueError(
                    "Warm-up patch schedule diverged from saved width-three source"
                )
            warm4[kind] = warm
            targets4[kind] = matched.training_targets(
                sub, "vanilla", warm["best"], f"targets4_{kind}_s{seed}"
            )
        seed_rows = []
        for arm, (rank, balanced, supervised, label) in ARMS.items():
            warm = warm3 if rank == 3 else warm4["mask" if supervised else "vanilla"]
            targets = (
                targets3 if rank == 3 else targets4["mask" if supervised else "vanilla"]
            )
            if arm == "mask3_fixed":
                phase1 = memory.required(
                    ctx["sources"]["large"], f"large_mask_ggf_s{seed}"
                )
                phase2 = memory.required(
                    ctx["sources"]["recovery"], f"recovery_s{seed}_control"
                )
                key = f"recovery_s{seed}_control"
                origin = "reused"
            else:
                armctx = subcontext(ctx, rank)
                phase1 = fit(
                    armctx,
                    f"phase1_{arm}_s{seed}",
                    warm["best"],
                    targets,
                    2500,
                    71000 + seed,
                    balanced,
                    supervised,
                    reference_mse=ref,
                )
                key = f"phase2_{arm}_s{seed}"
                # Reset model Adam at the same 2500 boundary as the saved control;
                # retain trained heads and the balancing controller's history.
                phase2 = fit(
                    armctx,
                    key,
                    phase1["weights"],
                    targets,
                    500,
                    82000 + seed,
                    balanced,
                    supervised,
                    reference_mse=ref,
                    initial_head=phase1["head"],
                    initial_balance=phase1["balance"],
                )
                origin = "new"
            if phase1["updates"] != 2500 or phase2["updates"] != 500:
                raise ValueError("Incomplete continuation")
            seed_rows.append(
                dict(
                    seed=seed,
                    arm=arm,
                    label=label,
                    rank=rank,
                    balanced=balanced,
                    supervised=supervised,
                    origin=origin,
                    key=key,
                    weights_sha=training.weight_sha(phase2["weights"]),
                    targets=targets,
                    targets_sha=identity(targets),
                    head_sha=training.weight_sha({"head": phase2["head"]})
                    if supervised
                    else None,
                    core_parameters=sum(v.numel() for v in phase2["weights"].values()),
                    head_parameters=150 if supervised else 0,
                    maximum_active_coordinates=16 * rank,
                    reference_mse=ref,
                    warm_selected_update=warm["best_update"],
                    phase1_batch_sha=phase1["sampled_batches_sha"],
                    phase2_batch_sha=phase2["sampled_batches_sha"],
                    phase1_noise_sha=phase1.get("sampled_noise_sha"),
                    phase2_noise_sha=phase2.get("sampled_noise_sha"),
                    new_gradient_updates=0 if origin == "reused" else 3000,
                    guided_seconds=phase1["seconds"] + phase2["seconds"],
                    warm_seconds=warm["seconds"],
                )
            )
            del phase1, phase2
            gc.collect()
            torch.cuda.empty_cache()
        for field in ["phase1_batch_sha", "phase2_batch_sha"]:
            if len({r[field] for r in seed_rows}) != 1:
                raise ValueError("Continuation sampling diverged: " + field)
        for field in ["phase1_noise_sha", "phase2_noise_sha"]:
            if len({r[field] for r in seed_rows if r["origin"] == "new"}) != 1:
                raise ValueError("New-arm noise stream diverged: " + field)
        rows.extend(seed_rows)
        s.write(f"paired_schedule_s{seed}.json", seed_rows)
        del warm3, warm4, mask_initial, random_initial
        gc.collect()
        torch.cuda.empty_cache()
    frozen = dict(
        complete=True,
        models=rows,
        new_gradient_updates=6 * 7640 + 12 * 3000,
        new_warmups=6,
        reused_controls=3,
        test_previously_inspected=True,
        structural_exchanges=0,
    )
    verify_catalog(ctx, frozen)
    matched.assert_sources(ctx)
    s.save("all_models_frozen", frozen)
    s.write("all_models_frozen.json", frozen)
    return frozen


def audit_one(ctx, row):
    s = ctx["store"]
    tag = f"s{row['seed']}_{row['arm']}"
    signature = identity(
        dict(
            weights=row["weights_sha"],
            targets=row["targets_sha"],
            rank=row["rank"],
            test=ctx["test_stamp"],
            cfg=ctx["cfg"],
        )
    )
    old = s.load("audit_" + tag)
    if old is not None:
        if old["identity"] != signature:
            raise ValueError("Cached audit identity changed")
        return old["report"]
    m = core.new_model(
        "vanilla",
        subcontext(ctx, row["rank"])["base"],
        get_weights(ctx, row),
        ctx["device"],
    )
    s.event("test_scoring_start", model=tag, images=1500)
    with matched.progress(ctx, m, tag):
        report = core.validation(
            m,
            ctx["packs"]["test"],
            row["targets"],
            ctx["cfg"],
            ctx["device"],
            with_arrays=True,
        )
        compact = study.compact_audit(
            report.pop("_audit"), matched.mapping(row["targets"]), ctx["cfg"]
        )
        activity = matched.activity(
            m, ctx["packs"]["test"], ctx["device"], ctx["cfg"]["batch"]
        )
    if len(report["categories"]) != 75 or any(
        not np.isfinite([r["capture"], r["ap"]]).all() for r in report["categories"]
    ):
        raise ValueError(
            "Test cohort cannot shrink or contain undefined category scores"
        )
    dev = training.measure(
        m,
        ctx["packs"]["select"]["x"].reshape(-1, 768),
        ctx["device"],
        ctx["cfg"]["batch"],
    )
    if not np.isfinite(
        [
            report["macro_capture"],
            report["macro_ap"],
            report["reconstruction"]["mse"],
            report["reconstruction"]["ve"],
            dev["mse"],
            dev["ve"],
        ]
    ).all():
        raise FloatingPointError(
            "Nonfinite final metric cannot become a completed result"
        )
    report.update(
        {
            k: row[k]
            for k in [
                "seed",
                "arm",
                "label",
                "rank",
                "origin",
                "core_parameters",
                "head_parameters",
                "maximum_active_coordinates",
                "weights_sha",
                "targets_sha",
            ]
        }
    )
    report.update(
        activity=activity,
        development_reconstruction=dev,
        common_native_MSE_limit=row["reference_mse"] * 1.02,
        legacy_native_eligible=bool(dev["mse"] <= row["reference_mse"] * 1.02),
    )
    if row["arm"] == "mask3_fixed":
        expected = memory.required(ctx["seed_stores"][row["seed"]], "fresh_result")[
            "ggf"
        ]
        if (
            max(
                abs(report["macro_capture"] - expected["macro_capture"]),
                abs(report["macro_ap"] - expected["macro_ap"]),
                abs(report["reconstruction"]["ve"] - expected["reconstruction"]["ve"]),
            )
            > 1e-5
        ):
            raise ValueError(
                "Reused control scores do not reproduce the saved common-test audit"
            )
    s.save("compact_" + tag, compact)
    s.save("audit_" + tag, dict(identity=signature, report=report))
    s.write("reports/" + tag + ".json", json_safe(report))
    folder = s.results / "image_contrasts"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(
        folder / (tag + ".npz"),
        image_ids=np.asarray(compact["image_ids"]),
        category_indices=np.asarray(compact["category_indices"]),
        denominator=compact["denominator"],
        error=compact["error"],
    )
    s.event(
        "test_scoring_complete",
        model=tag,
        capture=report["macro_capture"],
        AP=report["macro_ap"],
        VE=report["reconstruction"]["ve"],
    )
    del m, compact
    gc.collect()
    torch.cuda.empty_cache()
    return report


def evaluate(ctx):
    frozen = memory.required(ctx["store"], "all_models_frozen")
    verify_catalog(ctx, frozen)
    if "test" not in ctx["packs"]:
        raise RuntimeError("Load the cached test only after all models freeze")
    reports = [audit_one(ctx, r) for r in frozen["models"]]
    pairs = [
        ("mask4_fixed", "mask3_fixed"),
        ("mask3_balanced", "mask3_fixed"),
        ("mask4_balanced", "mask4_fixed"),
        ("mask4_balanced", "mask3_balanced"),
        ("mask4_balanced", "mask3_fixed"),
        ("mask4_fixed", "vanilla4"),
        ("mask4_balanced", "vanilla4"),
    ]
    comparisons = []
    s = ctx["store"]
    for seed in SEEDS:
        lookup = {r["arm"]: r for r in reports if r["seed"] == seed}
        for treatment, control in pairs:
            tag = f"s{seed}_{treatment}_vs_{control}"
            ci = s.load("paired_" + tag)
            if ci is None:
                s.event("paired_intervals_start", comparison=tag, resamples=200)
                a = memory.required(s, f"compact_s{seed}_{treatment}")
                b = memory.required(s, f"compact_s{seed}_{control}")
                ci = study.paired(a, b, ctx["cfg"])
                s.save("paired_" + tag, ci)
                del a, b
                gc.collect()
            a, b = lookup[treatment], lookup[control]
            comparisons.append(
                dict(
                    seed=seed,
                    treatment=treatment,
                    control=control,
                    capture_gain=a["macro_capture"] - b["macro_capture"],
                    ap_gain=a["macro_ap"] - b["macro_ap"],
                    ve_gain=a["reconstruction"]["ve"] - b["reconstruction"]["ve"],
                    paired_intervals=ci,
                    all_three_point_estimates_improve=bool(
                        a["macro_capture"] > b["macro_capture"]
                        and a["macro_ap"] > b["macro_ap"]
                        and a["reconstruction"]["ve"] > b["reconstruction"]["ve"]
                    ),
                    reconstruction_guard_passed=a["legacy_native_eligible"],
                )
            )
            s.write("paired/" + tag + ".json", json_safe(comparisons[-1]))
    out = dict(
        complete=True,
        models=reports,
        comparisons=comparisons,
        test_previously_inspected=True,
        no_multiple_comparison_correction=True,
        balanced_method="Bounded reconstruction-anchored GradNorm-inspired weighting; not canonical GradNorm",
        capacity="Width four adds 33.3% decoder/encoder capacity and changes 48 to 64 active slots; Vanilla4 matches width-four capacity",
        endpoint_rule=ctx["recipe"]["endpoint_rule"],
        new_structural_exchanges=0,
    )
    tables_and_plots(ctx, out)
    s.save("experiment_complete", out)
    s.write("summary.json", json_safe(out))
    matched.assert_sources(ctx)
    return out


def tables_and_plots(ctx, out):
    rows = []
    categories = []
    s = ctx["store"]
    for r in out["models"]:
        rows.append(
            dict(
                seed=r["seed"],
                model=r["label"],
                arm=r["arm"],
                rank=r["rank"],
                origin=r["origin"],
                capture_pct=100 * r["macro_capture"],
                patch_AP_pct=100 * r["macro_ap"],
                VE_pct=100 * r["reconstruction"]["ve"],
                MSE=r["reconstruction"]["mse"],
                core_parameters=r["core_parameters"],
                head_parameters=r["head_parameters"],
                maximum_active_coordinates=r["maximum_active_coordinates"],
                native_eligible=r["legacy_native_eligible"],
                mean_active_groups=r["activity"]["mean_active_groups"],
            )
        )
        categories.extend(
            dict(seed=r["seed"], arm=r["arm"], **c) for c in r["categories"]
        )
    matched.csv_write(s.results / "comparison.csv", rows)
    matched.csv_write(s.results / "all_categories.csv", categories)
    means = []
    for arm in ARMS:
        group = [r for r in rows if r["arm"] == arm]
        means.append(
            dict(
                arm=arm,
                model=ARMS[arm][3],
                capture_pct=float(np.mean([r["capture_pct"] for r in group])),
                patch_AP_pct=float(np.mean([r["patch_AP_pct"] for r in group])),
                VE_pct=float(np.mean([r["VE_pct"] for r in group])),
                eligible_seeds=sum(r["native_eligible"] for r in group),
            )
        )
    matched.csv_write(s.results / "three_seed_means.csv", means)
    paired = []
    for r in out["comparisons"]:
        ci = r["paired_intervals"]
        paired.append(
            dict(
                seed=r["seed"],
                treatment=r["treatment"],
                control=r["control"],
                capture_gain_pp=100 * r["capture_gain"],
                AP_gain_pp=100 * r["ap_gain"],
                VE_gain_pp=100 * r["ve_gain"],
                capture_ci_low=100 * ci["capture_gain_ci95"][0],
                capture_ci_high=100 * ci["capture_gain_ci95"][1],
                AP_ci_low=100 * ci["ap_gain_ci95"][0],
                AP_ci_high=100 * ci["ap_gain_ci95"][1],
                all_three_point_estimates_improve=r[
                    "all_three_point_estimates_improve"
                ],
                reconstruction_guard_passed=r["reconstruction_guard_passed"],
            )
        )
    matched.csv_write(s.results / "paired_comparison.csv", paired)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = s.results / "figures"
    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    colors = ["#879fb0", "#537cab", "#bc9bd0", "#008779", "#d4a86a"]
    for ax, key, title in zip(
        axes,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Object contrast capture", "Patch detection AP", "Reconstruction VE"],
    ):
        for j, r in enumerate(means):
            ax.bar(j, r[key], color=colors[j])
            ax.scatter(
                [j - 0.08, j, j + 0.08],
                [x[key] for x in rows if x["arm"] == r["arm"]],
                s=14,
                color="#222",
            )
            ax.text(j, r[key] + 1, f"{r[key]:.2f}%", ha="center", fontsize=9)
        ax.set_xticks(
            range(5),
            [
                "Mask 3\nFixed",
                "Mask 4\nFixed",
                "Mask 3\nBalanced",
                "Mask 4\nBalanced",
                "Vanilla 4",
            ],
            fontsize=9,
        )
        ax.set_ylim(
            min(0, min(r[key] for r in means)),
            max(100, max(r[key] for r in means) * 1.15),
        )
        ax.set_ylabel("%")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("Five arms, three seeds, same previously inspected 1,500-image test")
    for ext in ["png", "pdf"]:
        fig.savefig(folder / ("width_balance." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def export(ctx, source_path):
    result = matched.export(ctx, source_path)
    new = result.with_name(f"ggf_width_balance_{ctx['store'].run_id[:12]}_results.zip")
    result.replace(new)
    return new


def failure(ctx, stage, exc, source_path):
    import traceback

    try:
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                resume="Rerun unchanged. Mid-update failures replay the last complete verified checkpoint.",
            ),
        )
        print("Partial results:", export(ctx, source_path), flush=True)
    except BaseException as export_error:
        print(
            "Partial export failed:",
            type(export_error).__name__,
            str(export_error),
            flush=True,
        )


def remaining_updates(ctx):
    pending = {}
    s = ctx["store"]
    for kind in ["mask", "vanilla"]:
        remaining = 0
        for seed in SEEDS:
            state = s.load(f"warm4_{kind}_s{seed}")
            remaining += max(0, 7640 - (0 if state is None else state["updates"]))
            del state
        pending["warm4_" + kind] = remaining
    for arm in ARMS:
        if arm == "mask3_fixed":
            continue
        remaining = 0
        for seed in SEEDS:
            for phase, n in [(1, 2500), (2, 500)]:
                state = s.load(f"phase{phase}_{arm}_s{seed}")
                remaining += max(0, n - (0 if state is None else state["updates"]))
                del state
        pending[arm] = remaining
    return pending


def preflight(ctx):
    """Disposable full-size GPU steps; saved source weights and RNG stay intact."""
    device = ctx["device"]
    s = ctx["store"]
    cfg = ctx["cfg"]
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA GPU runtime before starting this study")
    hardware = dict(
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, device=device
    )
    old = s.load("preflight")
    if old is not None and old["hardware"] == hardware:
        timings = old["timings"]
    else:
        timings = {}
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        try:
            D = fourth_directions(ctx)
            initial4, random4 = initialize(subcontext(ctx, 4)["base"], D, 21)
            initial3 = memory.required(ctx["sources"]["large"], "large_mask_warm_s21")[
                "best"
            ]
            targets = memory.required(ctx["sources"]["large"], "large_mask_targets_s21")
            pack = ctx["packs"]["train"]
            indices = np.arange(cfg["images_per_update"])
            x = torch.as_tensor(np.array(pack["x"][indices], copy=True), device=device)
            xx = x.reshape(-1, 768)
            for arm, (rank, balanced, supervised, _) in ARMS.items():
                sub = subcontext(ctx, rank)
                w = initial3 if rank == 3 else (initial4 if supervised else random4)
                m = core.new_model("vanilla", sub["base"], w, device)
                head = torch.nn.Parameter(torch.zeros(75, 2, device=device))
                opt = torch.optim.Adam(
                    list(m.parameters()) + ([head] if supervised else []), lr=cfg["lr"]
                )
                controller = (
                    Balancer(ctx["recipe"]["balance"], device) if balanced else None
                )
                times = []
                diagnostics = None
                torch.cuda.reset_peak_memory_stats()
                for i in range(12):
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    rec, _ = m.loss(
                        xx + cfg["snr"] * xx.std() * torch.randn_like(xx), target=xx
                    )
                    cap = det = rec * 0
                    if supervised:
                        cap, det, _ = core.supervised_loss(
                            m,
                            x,
                            [pack["masks"][j] for j in indices],
                            targets,
                            head,
                            cfg,
                        )
                    losses = [rec, cap, det]
                    if not all(torch.isfinite(v) for v in losses):
                        raise FloatingPointError("Preflight loss is not finite")
                    used = (
                        controller.weights().detach()
                        if balanced
                        else rec.new_tensor([1.0, 0.05, 0.01])
                    )
                    if balanced:
                        # Exercise adaptation during this short disposable benchmark.
                        if controller.reference is None:
                            controller.reference = np.array(
                                [float(v.detach()) for v in losses]
                            )
                        if (i + 1) % 10 == 0:
                            norms, cosines = encoder_diagnostics(m, losses)
                            diagnostics = dict(
                                norms=norms.cpu().tolist(),
                                cosines=cosines,
                                adaptation=controller.adapt(losses, norms),
                            )
                    opt.zero_grad()
                    (
                        rec + used[1] * cap + used[2] * det if supervised else rec
                    ).backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        list(m.parameters()) + ([head] if supervised else []),
                        cfg["grad_clip"],
                    )
                    if not torch.isfinite(norm):
                        raise FloatingPointError("Preflight gradient is not finite")
                    opt.step()
                    m.normalize_decoder()
                    torch.cuda.synchronize()
                    if i >= 2:
                        times.append(time.monotonic() - started)
                with torch.no_grad():
                    _, z = m(xx)
                    if (z.norm(dim=-1) > 1e-6).sum(-1).max() > 16:
                        raise ValueError("Gate exceeded Top-16")
                timings[arm] = dict(
                    seconds_per_update=float(np.mean(times)),
                    peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                    weights=None
                    if controller is None
                    else controller.weights().detach().cpu().tolist(),
                    diagnostics=diagnostics,
                )
                s.event("preflight_arm_complete", arm=arm, **timings[arm])
                del m, head, opt, controller, z
                gc.collect()
                torch.cuda.empty_cache()
            # Warm-up has a different batch size and no supervised objectives.
            m = core.new_model("vanilla", subcontext(ctx, 4)["base"], random4, device)
            opt = torch.optim.Adam(m.parameters(), lr=ctx["base"]["lrs"][0])
            times = []
            xb = torch.as_tensor(
                np.array(pack["x"].reshape(-1, 768)[: ctx["base"]["batch"]], copy=True),
                device=device,
            )
            for i in range(8):
                torch.cuda.synchronize()
                started = time.monotonic()
                rec, _ = m.loss(
                    xb + ctx["base"]["snr"] * xb.std() * torch.randn_like(xb), target=xb
                )
                if not torch.isfinite(rec):
                    raise FloatingPointError("Nonfinite warm-up preflight loss")
                opt.zero_grad()
                rec.backward()
                if any(
                    not torch.isfinite(p.grad).all()
                    for p in m.parameters()
                    if p.grad is not None
                ):
                    raise FloatingPointError("Nonfinite warm-up preflight gradient")
                opt.step()
                m.normalize_decoder()
                torch.cuda.synchronize()
                if i >= 2:
                    times.append(time.monotonic() - started)
            timings["warm4"] = dict(seconds_per_update=float(np.mean(times)))
            del m, opt, x, xx, xb, initial3, initial4, random4
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            torch.set_rng_state(torch_state)
            torch.cuda.set_rng_state_all(cuda_state)
    pending = remaining_updates(ctx)
    seconds = sum(
        n
        * timings["warm4" if name.startswith("warm4_") else name]["seconds_per_update"]
        for name, n in pending.items()
    )
    out = dict(
        hardware=hardware,
        timings=timings,
        pending_updates=pending,
        estimated_gradient_hours=seconds / 3600,
        numerical_finite=True,
        note="Gradient time only. Fourth-direction eigendecompositions, development audits, training mappings, Drive saves, final scoring and 21 paired bootstrap comparisons add time.",
    )
    s.save("preflight", out)
    s.write("preflight.json", out)
    matched.assert_sources(ctx)
    return out


def load_test(ctx):
    frozen = memory.required(ctx["store"], "all_models_frozen")
    verify_catalog(ctx, frozen)
    source = ctx["sources"]["strong"]
    ids = ctx["test_ids"]
    stamp = ctx["test_stamp"]
    norm = ctx["norm"]
    folder = ctx["local"] / "normalized"
    folder.mkdir(exist_ok=True)
    path = folder / ("final_test_" + stamp["normalized_sha"] + ".npy")
    meta = path.with_suffix(".json")
    valid = (
        path.exists()
        and meta.exists()
        and json.loads(meta.read_text()).get("file_sha") == sha_file(path)
    )
    target = None
    if not valid:
        target = np.lib.format.open_memmap(
            path.with_suffix(".pending.npy"),
            mode="w+",
            dtype=np.float32,
            shape=(1500, 196, 768),
        )
    masks = []
    for lo in range(0, len(ids), ctx["test_chunk"]):
        shard = memory.required(source, f"features_test_{lo:06d}")
        want = ids[lo : lo + ctx["test_chunk"]]
        if (
            shard["ids"] != want
            or shard["normalization_sha"] != identity(norm)
            or len(shard["masks"]) != len(want)
        ):
            raise ValueError("Final cached shard identity changed")
        raw = (
            shard["x"].float().numpy()
            if torch.is_tensor(shard["x"])
            else np.asarray(shard["x"], np.float32)
        )
        if raw.shape != (len(want), 196, 768) or not np.isfinite(raw).all():
            raise ValueError("Invalid final cached activations")
        if target is not None:
            target[lo : lo + len(want)] = data.normalize(raw, norm)
        masks.extend(shard["masks"])
        del shard, raw
        if lo % 200 == 0:
            ctx["store"].event(
                "loading_cached_test", completed=lo + len(want), total=1500
            )
    if target is not None:
        target.flush()
        del target
        import os

        os.replace(path.with_suffix(".pending.npy"), path)
    x = np.load(path, mmap_mode="r")
    if (
        x.shape != (1500, 196, 768)
        or x.dtype != np.float32
        or memory.streaming_array_sha(x) != stamp["normalized_sha"]
    ):
        raise ValueError("Final test does not match the saved AGF activation identity")
    meta.write_text(
        json.dumps(
            dict(file_sha=sha_file(path), normalized_sha=stamp["normalized_sha"])
        )
    )
    ctx["packs"]["test"] = dict(x=x, ids=ids, masks=masks)
    matched.assert_sources(ctx)
    return ctx["packs"]["test"]
