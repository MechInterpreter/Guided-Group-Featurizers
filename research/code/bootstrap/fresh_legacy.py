"""Rebuild the width and continuation ancestor graph from public COCO inputs."""

import argparse
import copy
import gc
from pathlib import Path

from historical_inputs import (
    activate,
    audit_pack,
    coco_context,
    open_study,
    own,
    read,
    separate,
    sha,
    write,
)


DEPENDENCIES = {
    "native": (),
    "mask3": ("native",),
    "recovery3": ("native", "mask3"),
    "width4": ("native", "mask3", "recovery3"),
    "cap040": ("native", "width4"),
    "continue4": ("width4", "cap040"),
    "width5": ("width4", "cap040", "continue4"),
    "continue5": ("width4", "cap040", "continue4", "width5"),
    "capacity8": ("continue5",),
    "capacity12": ("capacity8",),
    "baselines3": ("native", "mask3", "recovery3"),
    "dictionary": ("native",),
    "direction_budget": ("native",),
}
COUNTS = dict(
    native=12,
    mask3=9,
    recovery3=18,
    width4=15,
    cap040=12,
    continue4=9,
    width5=15,
    continue5=9,
    capacity8=24,
    capacity12=12,
    baselines3=15,
    dictionary=2,
    direction_budget=6,
)


def recorded_configuration(code):
    record = read(code / "evidence/cap040_test/manifest.json")["config"]
    base = copy.deepcopy(record["base"])
    base.update(block_rank=3, local_save_updates=100)
    cfg = copy.deepcopy(record["fit_config"])
    return base, cfg


def context(root, features, code, stage):
    own(root, code, [features], __file__, "legacy-studies")
    activate(code, "components")
    from coco_coordinated import run as memory
    from coco_baselines import training
    from coco_baselines.storage import identity

    stores, catalogs = {}, {}
    for parent in DEPENDENCIES[stage]:
        receipt = read(root / parent / "FITS_COMPLETE.json")
        if not receipt.get("complete") or receipt.get("stage") != parent:
            raise ValueError("Finish the fresh prerequisite: " + parent)
        stores[parent] = open_study(
            root / parent, root / "scratch_reads" / stage / parent
        )
        catalogs[parent] = receipt["models"]
    base, cfg = recorded_configuration(code)
    if stage in [
        "cap040",
        "continue4",
        "width5",
        "continue5",
        "capacity8",
        "capacity12",
    ]:
        base.update(block_rank=4, local_save_updates=50)
    protocol = dict(
        stage=stage,
        native_config=base,
        fit_config=cfg,
        mapping_allocation_sha256=sha(code / "bootstrap/LEGACY_ALLOCATIONS.json"),
        dependencies={name: read(root / name / "FRESH_STUDY.json") for name in stores},
    )
    ctx = coco_context(root / stage, features, code, "components", __file__, protocol)
    ctx.update(
        base=base,
        cfg=cfg,
        device=cfg["device"],
        legacy_stores=stores,
        legacy_catalogs=catalogs,
        stage=stage,
        legacy_root=root,
        categories=cfg["category_indices"],
    )
    ctx["store"].write("provenance.json", ctx["provenance"])
    memory.memory_safe_metrics()
    for name, store in stores.items():
        if store.config["training_identity"] != ctx["packs"]["train"][
            "identity"
        ] or store.config["normalization_sha"] != identity(ctx["norm"]):
            raise ValueError(
                "Historical prerequisite uses another feature cohort: " + name
            )
        for row in catalogs[name]:
            source = row_store(ctx, row)
            st = memory.required(source, row["key"])
            if (
                training.weight_sha(st[row.get("field", "weights")])
                != row["weights_sha"]
                or st["updates"] != row["checkpoint_updates"]
            ):
                raise ValueError("Historical prerequisite weights changed")
            del st
    return ctx


def row_store(ctx, row):
    from coco_coordinated.run import FrozenStore

    if "fresh_store_root" not in row:
        raise ValueError("Only explicitly recorded fresh checkpoint stores are allowed")
    path = Path(row["fresh_store_root"]).resolve()
    if not path.is_relative_to(ctx["legacy_root"].resolve()):
        raise ValueError("Checkpoint store leaves this reproduction run")
    return FrozenStore(path, ctx["local"] / "read_endpoints" / path.name)


def normalize_rows(ctx, rows, sources=(), rank=3, total=10640):
    from coco_baselines import training
    from coco_baselines.storage import identity
    from coco_coordinated.run import required

    normalized = []
    candidates = [ctx["store"], *sources]
    for original in rows:
        row = copy.deepcopy(original)
        field = row.get("field", "weights")
        found = []
        inherited = [row_store(ctx, row)] if row.get("fresh_store_root") else []
        for source in candidates + inherited:
            state = source.load(row["key"])
            if (
                state is not None
                and field in state
                and state[field] is not None
                and training.weight_sha(state[field]) == row["weights_sha"]
            ):
                found.append((source, state))
            else:
                del state
        if len({str(source.root) for source, _ in found}) != 1:
            raise ValueError("Missing or ambiguous fresh checkpoint: " + row["key"])
        source, state = found[0]
        row.update(
            fresh_store_root=str(source.root),
            field=field,
            checkpoint_updates=state["updates"],
            rank=row.get("rank", rank),
            groups=row.get("groups", 1024),
            active_groups=row.get("active_groups", 16),
            total_training_updates=row.get(
                "total_training_updates",
                7640
                if state["updates"] == 7640
                else 10140
                if state["updates"] == 2500
                else total,
            ),
        )
        if ctx["stage"] == "dictionary":
            row.update(
                targets=None,
                targets_sha=None,
                mapping_status="Fit on the separate mapping cohort after model freeze",
            )
            normalized.append(row)
            del state, found
            continue
        if "targets" not in row:
            target_key = (
                f"large_mask_targets_s{row['seed']}"
                if row["key"].startswith("large_mask")
                else f"targets_s{row['seed']}"
            )
            row["targets"] = required(source, target_key)
        if [t["category_index"] for t in row["targets"]] != ctx["categories"]:
            raise ValueError("Historical category denominator changed")
        row["targets_sha"] = identity(row["targets"])
        row.setdefault("method", "vanilla")
        normalized.append(row)
        del state, found
        gc.collect()
    return normalized


def native(ctx):
    from coco_ggf import study, slics_closeout

    result = study.train(ctx)
    ctx["comparison"] = dict(rank=3)
    for seed in [21, 22, 23]:
        slics_closeout.init_slics(ctx, seed, "mask")
    rows = copy.deepcopy(result["models"])
    for row in rows:
        row["total_training_updates"] = 7640 if row["arm"] == "native" else 10140
    return normalize_rows(ctx, rows)


def mask3(ctx):
    from coco_ggf import large_mask
    from coco_coordinated.run import required

    source = ctx["legacy_stores"]["native"]
    ctx.update(
        source=source,
        comparison_source=source,
        source_frozen=required(source, "all_models_frozen"),
        large=large_mask.settings(),
    )
    result = large_mask.train(ctx)
    return normalize_rows(ctx, result["models"], [source], total=10140)


def recovery3(ctx):
    from coco_ggf import mask_recovery
    from coco_baselines import training
    from coco_coordinated.run import required

    source, parent = ctx["legacy_stores"]["native"], ctx["legacy_stores"]["mask3"]
    recipe = mask_recovery.settings()
    ctx.update(
        source=source,
        source_frozen=required(source, "all_models_frozen"),
        parent_mask=parent,
        recovery=recipe,
        cfg=dict(ctx["cfg"], updates=500, evaluate_every=50, diagnostic_grad_every=50),
        endpoints=[],
    )
    for seed in recipe["seeds"]:
        st = required(parent, f"large_mask_ggf_s{seed}")
        ctx["endpoints"].append(
            dict(
                seed=seed,
                weights_sha=training.weight_sha(st["weights"]),
                head_sha=training.weight_sha({"head": st["head"]}),
            )
        )
        del st
    result = mask_recovery.train(ctx)
    return normalize_rows(ctx, result["models"], [source, parent])


def width4(ctx):
    from coco_ggf import width_balance
    from coco_coordinated.run import required

    native_store = ctx["legacy_stores"]["native"]
    ctx.update(
        sources=dict(
            large=ctx["legacy_stores"]["mask3"],
            recovery=ctx["legacy_stores"]["recovery3"],
        ),
        recipe=width_balance.settings(),
        reused=[],
    )
    for seed in [21, 22, 23]:
        st = required(native_store, f"ggf_s{seed}")
        ctx["reused"].append(dict(seed=seed, native_reference_mse=st["initial_mse"]))
        del st
    result = width_balance.freeze(ctx)
    return normalize_rows(ctx, result["models"], list(ctx["sources"].values()), rank=4)


def cap040(ctx):
    from coco_ggf import weight_bound
    from coco_baselines import training
    from coco_coordinated.run import required

    parent = ctx["legacy_stores"]["width4"]
    previous = ctx["legacy_catalogs"]["width4"]
    ctx.update(
        parent=parent,
        sources=dict(original=ctx["legacy_stores"]["native"]),
        catalog={(r["seed"], r["arm"]): r for r in previous},
        recipe=weight_bound.settings(),
        seeds=[21, 22, 23],
        base=dict(ctx["base"], block_rank=4),
    )
    weight_bound.run(ctx)
    rows = []
    for row in previous:
        if row["arm"] == "mask4_balanced":
            rows.append(copy.deepcopy(row))
            st = required(ctx["store"], f"phase2_cap040_s{row['seed']}")
            row = dict(
                row,
                arm="cap040",
                key=f"phase2_cap040_s{row['seed']}",
                field="weights",
                weights_sha=training.weight_sha(st["weights"]),
                head_sha=training.weight_sha({"head": st["head"]}),
                development=st["history"][-1]["validation"],
            )
            del st
            rows.append(row)
        elif row["arm"] in ["vanilla4", "mask4_balanced", "mask3_fixed"]:
            rows.append(copy.deepcopy(row))
    return normalize_rows(
        ctx,
        rows,
        [parent, ctx["legacy_stores"]["native"]]
        + list(ctx.get("legacy_stores", {}).values()),
        rank=4,
    )


def continue4(ctx):
    from coco_ggf import pcgrad_continuation as study

    roots = copy.deepcopy(
        [
            r
            for r in ctx["legacy_catalogs"]["cap040"]
            if r["arm"] in ["cap040", "vanilla4"]
        ]
    )
    for row in roots:
        row["source_name"] = "width_balance" if row["arm"] == "vanilla4" else "cap040"
    ctx.update(
        roots=roots,
        sources=dict(
            width_balance=ctx["legacy_stores"]["width4"],
            cap040=ctx["legacy_stores"]["cap040"],
        ),
        cap040_rows=roots,
        base=dict(ctx["base"], block_rank=4),
        recipe=study.settings(),
    )
    result = study.run(ctx)
    return normalize_rows(ctx, result["models"], rank=4, total=13640)


def width5(ctx):
    from coco_ggf import next_experiments as study

    parent = ctx["legacy_stores"]["width4"]
    roots = copy.deepcopy(
        ctx["legacy_catalogs"]["cap040"] + ctx["legacy_catalogs"]["continue4"]
    )
    for row in roots:
        row["source_name"] = (
            "previous"
            if row["arm"] in ["ggf_continue", "ggf_pcgrad", "vanilla_continue"]
            else "cap040"
            if row["arm"] == "cap040"
            else "width_balance"
        )
    ctx.update(
        parent=parent,
        roots=roots,
        sources=dict(
            previous=ctx["legacy_stores"]["continue4"],
            cap040=ctx["legacy_stores"]["cap040"],
            width_balance=parent,
        ),
        base=dict(ctx["base"], block_rank=4),
        recipe=study.settings(),
    )
    rows = study.run_width5(ctx)
    # The other two declared 4D continuation controls use the same fitted roots.
    rows += study.run_continuation(ctx, "pcgrad_both")
    rows += study.run_continuation(ctx, "longer")
    for row in rows:
        if row["arm"] not in ["mask5", "vanilla5"]:
            row["rank"] = 4
    return normalize_rows(ctx, rows, rank=5)


def continue5(ctx):
    from coco_ggf import five_continuation as study

    roots = [
        r for r in ctx["legacy_catalogs"]["width5"] if r["arm"] in ["mask5", "vanilla5"]
    ]
    roots += ctx["legacy_catalogs"]["continue4"]
    ctx.update(
        roots=roots,
        parent=ctx["legacy_stores"]["width4"],
        sources=dict(
            width5=ctx["legacy_stores"]["width5"], cap040=ctx["legacy_stores"]["cap040"]
        ),
        cap040_rows=ctx["legacy_catalogs"]["cap040"],
        base=dict(ctx["base"], block_rank=4),
        recipe=study.settings(),
    )
    result = study.run(ctx)
    return normalize_rows(ctx, result["models"], rank=5, total=13640)


def capacity(ctx, width):
    if width == 8:
        from coco_ggf import warm_capacity as study

        parent_name, source_name = "continue5", "continued5"
    else:
        from coco_ggf import twelve_capacity as study

        parent_name, source_name = "capacity8", "capacity_parent"
    ctx.update(
        roots=ctx["legacy_catalogs"][parent_name],
        sources={source_name: ctx["legacy_stores"][parent_name]},
    )
    result = study.run(ctx)
    return normalize_rows(ctx, result["models"], rank=width, total=study.TOTAL)


def baselines3(ctx):
    """The five 3D methods retained in the paper, with the original kernels."""
    from coco_ggf import matched_baselines as study
    from coco_coordinated.run import required

    s = ctx["store"]
    native = ctx["legacy_stores"]["native"]
    masked = ctx["legacy_stores"]["mask3"]
    recovery = ctx["legacy_stores"]["recovery3"]
    x = ctx["packs"]["train"]["x"].reshape(-1, 768)
    val = ctx["packs"]["select"]["x"].reshape(-1, 768)
    rows = []
    for seed in [21, 22, 23]:
        reference = required(native, f"ggf_s{seed}")["initial_mse"]
        for arm in ["vanilla", "grassmannian", "group_lasso", "mask_reconstruction"]:
            method = arm if arm in ["grassmannian", "group_lasso"] else "vanilla"
            if arm == "vanilla":
                warm = required(native, f"native_s{seed}")
                phase1 = required(native, f"continued_s{seed}")
                targets = required(native, f"targets_s{seed}")
            else:
                if arm == "mask_reconstruction":
                    warm = required(masked, f"large_mask_warm_s{seed}")
                    targets = required(masked, f"large_mask_targets_s{seed}")
                else:
                    warm = study.fit_warm(
                        s,
                        f"warm_{arm}_s{seed}",
                        method,
                        x,
                        val,
                        dict(ctx["base"], local_save_updates=50),
                        seed,
                        ctx["base"]["lrs"][0],
                        ctx["device"],
                        drive_every=50,
                    )
                    start, _, _ = study.warm_choice(warm, method)
                    targets = study.training_targets(
                        ctx, method, start, f"targets_{arm}_s{seed}"
                    )
                start, _, _ = study.warm_choice(warm, method)
                phase1 = study.fit(
                    ctx, f"phase1_{arm}_s{seed}", method, start, 2500, 71000 + seed
                )
            phase2 = study.fit(
                ctx,
                f"phase2_{arm}_s{seed}",
                method,
                phase1["weights"],
                500,
                82000 + seed,
            )
            study.complete(phase2, 500)
            rows.append(
                study.catalog_row(
                    ctx,
                    seed,
                    arm,
                    phase2["weights"],
                    targets,
                    key=f"phase2_{arm}_s{seed}",
                    field="weights",
                    native_reference_mse=reference,
                    phase1_batch_sha=phase1["sampled_batches_sha"],
                    phase2_batch_sha=phase2["sampled_batches_sha"],
                    warm_selected_update=warm["best_update"],
                    structural_exchanges=0,
                )
            )
            del warm, phase1, phase2
            gc.collect()
        first = required(masked, f"large_mask_ggf_s{seed}")
        tail = required(recovery, f"recovery_s{seed}_control")
        targets = required(masked, f"large_mask_targets_s{seed}")
        rows.append(
            study.catalog_row(
                ctx,
                seed,
                "mask_ggf",
                tail["weights"],
                targets,
                key=f"recovery_s{seed}_control",
                field="weights",
                native_reference_mse=reference,
                phase1_batch_sha=first["sampled_batches_sha"],
                phase2_batch_sha=tail["sampled_batches_sha"],
                structural_exchanges=0,
            )
        )
        if (
            len({r["phase1_batch_sha"] for r in rows if r["seed"] == seed}) != 1
            or len({r["phase2_batch_sha"] for r in rows if r["seed"] == seed}) != 1
        ):
            raise ValueError("Matched 3D baseline image schedules differ")
    if len(rows) != 15:
        raise ValueError("Retain all five methods and three orders")
    return normalize_rows(ctx, rows, [recovery])


def dictionary(ctx):
    from coco_ggf import vanilla_width as study

    record = read(
        Path(__file__).resolve().parents[1]
        / "evidence/dictionary_size/agf_dictionary_size_results.json"
    )["provenance"]
    ctx.update(
        base=record["native_config"],
        audit=record["audit_config"],
        cfg=record["config"],
        source=ctx["legacy_stores"]["native"],
        policy=dict(reuse_1024=True),
        reference=next(
            r
            for r in ctx["legacy_catalogs"]["native"]
            if r["seed"] == 21 and r["arm"] == "native"
        ),
    )
    result = study.train(ctx)
    return normalize_rows(ctx, result["models"], [ctx["source"]], total=7640)


def direction_budget(ctx):
    from coco_ggf import wider_groups as study
    from coco_coordinated.run import required

    parent = ctx["legacy_stores"]["native"]
    recipe = study.settings()
    ctx.update(
        source=parent,
        targets={
            seed: required(parent, f"targets_s{seed}") for seed in recipe["seeds"]
        },
        sweep=recipe,
        grid=recipe["grid"],
    )
    summary = study.run(ctx)
    room = study.headroom(ctx)
    study.figures(ctx, summary, room)
    rows = []
    for entry in summary["models"]:
        key = entry["setting"] + "_s" + str(entry["seed"])
        targets = required(ctx["store"], "targets_" + key)
        for field, sha_key in [
            ("best", "selected_weights_sha"),
            ("weights", "final_weights_sha"),
        ]:
            rows.append(
                dict(
                    seed=entry["seed"],
                    arm=entry["setting"] + "_" + field,
                    key="ggf_" + key,
                    field=field,
                    weights_sha=entry[sha_key],
                    targets=targets,
                    groups=entry["groups"],
                    rank=entry["rank"],
                    active_groups=entry["active"],
                    total_training_updates=10140,
                )
            )
    return normalize_rows(ctx, rows, total=10140)


def train(ctx, root, stage):
    action = {
        "capacity8": lambda c: capacity(c, 8),
        "capacity12": lambda c: capacity(c, 12),
    }.get(stage, globals().get(stage))
    rows = action(ctx)
    validate_rows(rows, stage)
    write(
        root / stage / "FITS_COMPLETE.json",
        dict(
            complete=True,
            stage=stage,
            models=rows,
            feature_run=ctx["store"].config["feature_run"],
        ),
    )


def validate_rows(rows, stage):
    expected_seeds = (
        {21} if stage in ["dictionary", "direction_budget"] else {21, 22, 23}
    )
    if (
        len(rows) != COUNTS[stage]
        or {r["seed"] for r in rows} != expected_seeds
        or len(
            {
                (r["seed"], r["arm"], r["groups"], r["rank"], r["active_groups"])
                for r in rows
            }
        )
        != len(rows)
    ):
        raise ValueError("Incomplete or duplicate historical model catalog")


def evaluate(ctx, root, code, stage):
    from coco_adaptation import core
    from coco_baselines import training
    from coco_coordinated.run import required

    receipt = read(root / stage / "FITS_COMPLETE.json")
    if (
        not receipt.get("complete")
        or receipt["stage"] != stage
        or receipt["feature_run"] != ctx["store"].config["feature_run"]
    ):
        raise ValueError("Complete this exact historical stage before scoring")
    validate_rows(receipt["models"], stage)
    if stage == "direction_budget":
        summary = ctx["store"].load("wider_complete")
        if not summary or not summary.get("complete"):
            raise ValueError("Complete the development-only direction-budget screen")
        write(
            root / stage / "RESULTS_COMPLETE.json",
            dict(
                complete=True,
                stage=stage,
                results=summary,
                evaluation_split="selection",
            ),
        )
        return
    if stage == "dictionary":
        from coco_ggf import vanilla_width

        reference = read(
            code / "evidence/dictionary_size/agf_dictionary_size_results.json"
        )["provenance"]
        ctx.update(
            base=reference["native_config"],
            cfg=reference["config"],
            audit=reference["audit_config"],
            source=ctx["legacy_stores"]["native"],
            policy=dict(reuse_1024=True),
        )
        ctx["packs"]["mapping"] = audit_pack(
            ctx, code, root / stage, "legacy-mapping", receipt
        )
        summary = vanilla_width.evaluate(ctx)
        write(
            root / stage / "RESULTS_COMPLETE.json",
            dict(
                complete=True,
                stage=stage,
                results=summary,
                evaluation_split="confirmation",
            ),
        )
        return
    pack = audit_pack(ctx, code, root / stage, "controls", receipt)
    calibration = (
        audit_pack(ctx, code, root / stage, "legacy-mapping", receipt)
        if stage in ["native", "mask3", "recovery3"]
        else None
    )
    results = []
    for row in receipt["models"]:
        st = required(row_store(ctx, row), row["key"])
        weights = st[row["field"]]
        if (
            training.weight_sha(weights) != row["weights_sha"]
            or st["updates"] != row["checkpoint_updates"]
        ):
            raise ValueError("Frozen historical endpoint changed")
        base = dict(
            ctx["base"],
            groups=row["groups"],
            block_rank=row["rank"],
            active_groups=row["active_groups"],
        )
        model = core.new_model(row["method"], base, weights, ctx["device"])
        if calibration is not None:
            from coco_baselines import metrics
            import numpy as np

            observed = metrics.audit_arrays(
                model,
                calibration["x"],
                calibration["masks"],
                calibration["ids"],
                ctx["cfg"],
                ctx["device"],
            )
            fixed = metrics.calibrate(
                observed, ctx["cfg"], ctx["provenance"]["categories"]
            )
            del observed
            gc.collect()
            arrays = metrics.audit_arrays(
                model, pack["x"], pack["masks"], pack["ids"], ctx["cfg"], ctx["device"]
            )
            per = metrics.evaluate(
                arrays,
                fixed,
                dict(ctx["cfg"], bootstrap=0),
                ctx["cfg"]["fresh_test_seed"],
            )
            by = {r["category_index"]: r for r in per}
            selected = [by[c] for c in ctx["categories"]]
            if any(r["contrast"] is None or r["detection"] is None for r in selected):
                raise ValueError(
                    "Calibration readouts cannot drop unsupported categories"
                )
            report = dict(
                categories=selected,
                macro_capture=float(
                    np.mean([r["contrast"]["single_group_capture"] for r in selected])
                ),
                macro_ap=float(np.mean([r["detection"]["ap"] for r in selected])),
                reconstruction=training.measure(
                    model,
                    pack["x"].reshape(-1, model.d),
                    ctx["device"],
                    ctx["cfg"]["batch"],
                ),
                _audit=arrays,
            )
            ctx["store"].save("mapping_" + row["arm"] + "_s" + str(row["seed"]), fixed)
        else:
            report = core.validation(
                model, pack, row["targets"], ctx["cfg"], ctx["device"], with_arrays=True
            )
        if (
            len(report["categories"]) != 75
            or calibration is None
            and any(
                r["capture"] is None or r["ap"] is None for r in report["categories"]
            )
        ):
            raise ValueError("Retain every declared historical category")
        arrays = report.pop("_audit")
        from coco_ggf.study import compact_audit
        from coco_ggf.matched_baselines import mapping

        compact = compact_audit(
            arrays,
            fixed if calibration is not None else mapping(row["targets"]),
            ctx["cfg"],
        )
        ctx["store"].save("compact_" + row["arm"] + "_s" + str(row["seed"]), compact)
        results.append(
            dict(
                seed=row["seed"],
                arm=row["arm"],
                weights_sha=row["weights_sha"],
                **report,
            )
        )
        del model, st, weights, arrays, compact
        gc.collect()
    import numpy as np

    means = []
    for arm in sorted({r["arm"] for r in results}):
        rows = [r for r in results if r["arm"] == arm]
        result = dict(arm=arm, seeds=[r["seed"] for r in rows])
        for key, values in [
            ("capture", [r["macro_capture"] for r in rows]),
            ("patch_AP", [r["macro_ap"] for r in rows]),
            ("VE", [r["reconstruction"]["ve"] for r in rows]),
        ]:
            result[key] = dict(
                mean=float(np.mean(values)),
                sample_SD=float(np.std(values, ddof=1)) if len(values) > 1 else None,
            )
        means.append(result)
    comparisons = []
    from itertools import combinations
    from coco_ggf.study import paired

    for seed in sorted({r["seed"] for r in results}):
        arms = sorted(r["arm"] for r in results if r["seed"] == seed)
        for first, second in combinations(arms, 2):
            key = f"paired_s{seed}_{first}_vs_{second}"
            saved = ctx["store"].load(key)
            if saved is None:
                saved = paired(
                    required(ctx["store"], f"compact_{first}_s{seed}"),
                    required(ctx["store"], f"compact_{second}_s{seed}"),
                    ctx["cfg"],
                )
                ctx["store"].save(key, saved)
            comparisons.append(dict(seed=seed, first=first, second=second, **saved))
    write(
        root / stage / "RESULTS_COMPLETE.json",
        dict(
            complete=True,
            stage=stage,
            models=results,
            means=means,
            paired_comparisons=comparisons,
            audit_identity=pack["identity"],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--features-root", required=True, type=Path)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--stage", choices=DEPENDENCIES, required=True)
    p.add_argument("--phase", choices=["train", "evaluate"], required=True)
    a = p.parse_args()
    root, features, code = (
        a.root.resolve(),
        a.features_root.resolve(),
        a.code_root.resolve(),
    )
    separate(root, features, code)
    ctx = context(root, features, code, a.stage)
    train(ctx, root, a.stage) if a.phase == "train" else evaluate(
        ctx, root, code, a.stage
    )


if __name__ == "__main__":
    main()
