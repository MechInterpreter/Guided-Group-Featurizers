"""Three fixed-budget 5D continuation arms from verified, immutable roots."""

import copy, gc, json, traceback
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import next_experiments as prior, next_fit as engine

base = prior.base
matched = prior.matched
SEEDS = [21, 22, 23]
WIDTH5_RUN = "95fbab1d48bff58545543e390b89e28c87c8cc8612db4fc25e9759489407befb"
WIDTH5_SOURCE_SHA = "0c0a52cdea8d445de2f4edf32273ad388849a7c0824d49ee3381d7f6931bb105"
UPDATES = 3000
STAGE = "continuation5"
ARMS = ["guided5_continue", "vanilla5_continue", "pcgrad5_both"]
CONTROLS = [
    "mask5",
    "vanilla5",
    "ggf_continue",
    "ggf_pcgrad",
    "vanilla_continue",
    "cap040",
    "vanilla4",
]
STAGE_ARMS = {STAGE: ARMS}
STAGE_CONTROLS = {STAGE: CONTROLS}
PAIRS = {
    STAGE: [
        ("guided5_continue", "vanilla5_continue"),
        ("guided5_continue", "mask5"),
        ("vanilla5_continue", "vanilla5"),
        ("guided5_continue", "ggf_continue"),
        ("vanilla5_continue", "vanilla_continue"),
        ("pcgrad5_both", "guided5_continue"),
        ("pcgrad5_both", "vanilla5_continue"),
        ("pcgrad5_both", "mask5"),
        ("pcgrad5_both", "ggf_pcgrad"),
    ]
}
LABELS = dict(
    prior.LABELS,
    guided5_continue="AGF-G 5D / ordinary continuation",
    vanilla5_continue="Vanilla BSF 5D / continuation",
    pcgrad5_both="AGF-G 5D / encoder + decoder PCGrad",
)
subcontext = prior.subcontext
root_for = prior.root_for
audit = prior.audit


def settings():
    return dict(
        version=1,
        study="agf_five_dimensional_continuation",
        seeds=SEEDS,
        arms=ARMS,
        groups=1024,
        block_rank=5,
        active_groups=16,
        maximum_active_coordinates=80,
        balance=copy.deepcopy(base.bound.settings()["balance"]),
        updates_per_arm=UPDATES,
        seed_offset=93000,
        new_planned_updates=27000,
        roots="Same final 5D mask endpoint for both guided branches; paired final 5D Vanilla endpoint for control.",
        optimizer="Carry Adam moments, trained heads and controller. No reset or warm-up.",
        pcgrad="Separate weighted encoder R/C/D and decoder R/C projection domains; private heads ordinary.",
        controls="Reuse saved 4D ordinary, encoder-only PCGrad, Vanilla continuation and 4D/5D starting audits.",
        execution="Ordinary guided all seeds, Vanilla all seeds, then both-domain PCGrad all seeds. Freeze nine endpoints before test.",
        endpoint="Final 3000-update endpoints, scheduled total 13640. Retain original native-MSE x 1.02 failures.",
        test_previously_inspected=True,
        new_structural_exchanges=0,
        pcgrad_reference="https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf",
    )


def validate_catalog(frozen, result):
    expected = {(s, a) for s in SEEDS for a in ["mask5", "vanilla5"]}
    if (
        not frozen.get("complete")
        or not result.get("complete")
        or result.get("stage") != "width5"
        or len(frozen.get("models", [])) != 6
        or {(r["seed"], r["arm"]) for r in frozen["models"]} != expected
    ):
        raise ValueError(
            "Finish Section 4: all three 5D guided models, Vanilla controls, and saved width5 evaluation are required. No retraining fallback."
        )
    for row in frozen["models"]:
        if (
            row["rank"] != 5
            or row["total_training_updates"] != 10640
            or row["checkpoint_updates"] != 500
            or row["key"] != f"phase2_{row['arm']}_s{row['seed']}"
            or row["supervised"] != (row["arm"] == "mask5")
            or row["balanced"] != row["supervised"]
        ):
            raise ValueError(
                "Wrong 5D source capacity, budget, endpoint or supervision"
            )


def validate_width5(ctx, source, frozen, result):
    cfg = source.config
    if (
        source.run_id != WIDTH5_RUN
        or cfg.get("source_sha") != WIDTH5_SOURCE_SHA
        or cfg.get("continuation_recipe") != prior.settings()
        or cfg["base"] != ctx["base"]
        or cfg["fit_config"] != ctx["cfg"]
        or cfg["versions"] != matched.versions()
        or cfg["parent_run"] != ctx["parent"].run_id
        or cfg["cap040_run"] != ctx["sources"]["cap040"].run_id
        or cfg["previous_run"] != prior.PREVIOUS_RUN
    ):
        raise ValueError("5D source protocol, package versions, or lineage differs")
    validate_catalog(frozen, result)
    for row in frozen["models"]:
        matched.check_targets(row["targets"], ctx["cfg"])
        st = memory.required(source, row["key"])
        matched.complete(st, 500)
        check_state(ctx, row, st)
        old4 = root_for(ctx, row["seed"], "cap040")
        if row["reference_mse"] != old4["reference_mse"]:
            raise ValueError("5D source changed the common native reconstruction limit")
        tag = f"s{row['seed']}_{row['arm']}"
        saved = memory.required(source, "audit_" + tag)
        compact = memory.required(source, "compact_" + tag)
        if saved["identity"] != base.report_signature(
            row, ctx["test_stamp"], ctx["cfg"]
        ):
            raise ValueError("5D source audit uses a different test cohort or readout")
        base.check_report(saved["report"], row, ctx["cfg"])
        entries = [
            r
            for r in result["models"]
            if (r["seed"], r["arm"]) == (row["seed"], row["arm"])
        ]
        if len(entries) != 1 or identity(entries[0]) != identity(saved["report"]):
            raise ValueError("5D summary does not match its saved audit")
        if (
            compact["image_ids"] != ctx["test_ids"]
            or compact["category_indices"] != ctx["cfg"]["category_indices"]
        ):
            raise ValueError("5D compact audit cohort differs")
        for prefix, value in [("audit_", saved), ("compact_", compact)]:
            ctx["store"].save(prefix + tag, value)
        ctx["store"].write(
            "reports/" + tag + ".json", base.width.json_safe(saved["report"])
        )
        ctx["roots"].append(dict(row, source_name="width5"))
        ctx["checkpoint_stamps"].extend(
            matched.source_key_stamp(source, k)
            for k in [row["key"], "audit_" + tag, "compact_" + tag]
        )
        print(
            "Verified final 5D endpoint, optimizer/controller and cached audit:",
            tag,
            flush=True,
        )
        del st, saved, compact
        gc.collect()
    for seed in SEEDS:
        rows = [r for r in frozen["models"] if r["seed"] == seed]
        for field in [
            "warm_batch_sha",
            "warm_noise_sha",
            "phase1_batch_sha",
            "phase1_noise_sha",
            "phase_batch_sha",
            "phase_noise_sha",
        ]:
            if len({r[field] for r in rows}) != 1:
                raise ValueError("5D source arms have unequal image/noise schedules")
    ctx["checkpoint_stamps"].extend(
        matched.source_key_stamp(source, k) for k in ["frozen_width5", "result_width5"]
    )


def check_state(ctx, row, st):
    matched.valid_weights(st["weights"], "vanilla", subcontext(ctx, 5)["base"])
    if (
        training.weight_sha(st["weights"]) != row["weights_sha"]
        or identity(row["targets"]) != row["targets_sha"]
        or identity(engine.state_identity(st["optimizer"])) != row["optimizer_sha"]
        or identity(engine.state_identity(st["balance"])) != row["balance_sha"]
        or row["supervised"]
        and training.weight_sha({"head": st["head"]}) != row["head_sha"]
        or st["sampled_batches_sha"] != row["phase_batch_sha"]
        or st["sampled_noise_sha"] != row["phase_noise_sha"]
    ):
        raise ValueError(
            "5D checkpoint weights, readouts, optimizer/controller or sampling identity changed"
        )
    if row["supervised"]:
        weights = st["balance"]["log_weights"].detach().exp()
        if (
            weights.shape != (2,)
            or not torch.isfinite(weights).all()
            or not (
                0.0125 - 1e-7 <= weights[0] <= 0.4 + 1e-7
                and 0.0025 - 1e-7 <= weights[1] <= 0.4 + 1e-7
            )
        ):
            raise ValueError("Saved adaptive weights violate the cap-0.40 recipe")
    elif st["balance"] is not None:
        raise ValueError("Vanilla source has a supervised controller")


def prepare(
    paths,
    output,
    local,
    cache_local,
    source_sha,
    previous_output,
    width5_output,
    previous_test_output=None,
    require_drive=True,
):
    local = Path(local)
    base.safe_destination(
        output, list(paths.values()) + [previous_output, width5_output]
    )
    # Fail before loading large feature caches if Section 4 is still incomplete.
    source = matched.discover(width5_output, local / "read_width5", run=WIDTH5_RUN)
    frozen = source.load("frozen_width5")
    result = source.load("result_width5")
    if frozen is None or result is None:
        raise FileNotFoundError(
            "Section 4 has not completed and saved all three seeds plus controls and evaluations. Finish it, stop the previous notebook, then Run all here."
        )
    validate_catalog(frozen, result)
    # Inherited verified loaders read sources only and keep test features closed.
    ctx = prior.prepare(
        paths,
        local / "verification_five",
        local,
        cache_local,
        source_sha,
        previous_output,
        previous_test_output,
        False,
    )
    from .control_audit_reuse import restore

    restore(ctx)
    old = ctx["store"]
    payload = dict(
        old.config,
        study=settings()["study"],
        continuation_recipe=settings(),
        width5_run=WIDTH5_RUN,
        maximum_planned_gradient_updates=27000,
        endpoint="Final 3000-update continuation endpoints; no best-checkpoint selection.",
        test_access="Only after all nine 5D continuation endpoints freeze",
    )
    ctx["store"] = inputs.JournalStore(
        output, payload, local / "five_rescue", require_drive
    )
    base.install_drive_recovery(ctx["store"])
    ctx["recipe"] = settings()
    for seed in SEEDS:
        for arm in [
            "ggf_continue",
            "ggf_pcgrad",
            "vanilla_continue",
            "cap040",
            "vanilla4",
        ]:
            for prefix in ["audit_", "compact_"]:
                value = memory.required(old, f"{prefix}s{seed}_{arm}")
                ctx["store"].save(f"{prefix}s{seed}_{arm}", value)
                if prefix == "audit_":
                    ctx["store"].write(
                        f"reports/s{seed}_{arm}.json",
                        base.width.json_safe(value["report"]),
                    )
                del value
    ctx["sources"]["width5"] = source
    ctx["stamps"] += memory.source_snapshot([source])
    validate_width5(ctx, source, frozen, result)
    # Needed only by the inherited, separately verified test-feature loader.
    ctx["store"].save(
        "all_models_frozen", dict(complete=True, models=ctx["cap040_rows"])
    )
    ctx["store"].write("protocol.json", payload)
    ctx["store"].write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    ctx["store"].write("test_identity.json", ctx["test_stamp"])
    matched.assert_sources(ctx)
    return ctx


def source_state(ctx, row):
    st = memory.required(ctx["sources"]["width5"], row["key"])
    matched.complete(st, 500)
    check_state(ctx, row, st)
    return st


def fit_branch(ctx, key, state, root, seed, supervised, both=False, max_updates=None):
    # PyTorch may alias optimizer tensors when loading a controller on CPU.
    # Copy carried controller state so branching/resuming cannot mutate its root.
    return engine.fit(
        subcontext(ctx, 5),
        key,
        state["weights"],
        root["targets"],
        UPDATES,
        93000 + seed,
        supervised,
        supervised,
        root["reference_mse"],
        initial_head=state["head"] if supervised else None,
        initial_balance=copy.deepcopy(state["balance"]) if supervised else None,
        initial_optimizer=state["optimizer"],
        pcgrad=both,
        decoder_pcgrad=both,
        max_updates=max_updates,
    )


def endpoint_row(ctx, seed, arm, state, root):
    supervised = arm != "vanilla5_continue"
    matched.valid_weights(state["weights"], "vanilla", subcontext(ctx, 5)["base"])
    v = state["history"][-1]["validation"]
    if not supervised:
        model = core.new_model(
            "vanilla", subcontext(ctx, 5)["base"], state["weights"], ctx["device"]
        )
        with torch.inference_mode():
            v = core.validation(
                model,
                ctx["packs"]["select"],
                root["targets"],
                ctx["cfg"],
                ctx["device"],
            )
        del model
    base.bound.validate_metrics(v)
    return dict(
        seed=seed,
        arm=arm,
        label=LABELS[arm],
        rank=5,
        origin="continued_saved_5D_endpoint",
        targets=root["targets"],
        targets_sha=root["targets_sha"],
        key=f"{arm}_s{seed}",
        checkpoint_updates=UPDATES,
        weights_sha=training.weight_sha(state["weights"]),
        head_sha=training.weight_sha({"head": state["head"]}) if supervised else None,
        optimizer_sha=identity(engine.state_identity(state["optimizer"])),
        balance_sha=identity(engine.state_identity(state["balance"])),
        supervised=supervised,
        balanced=supervised,
        pcgrad=arm == "pcgrad5_both",
        decoder_pcgrad=arm == "pcgrad5_both",
        core_parameters=sum(v.numel() for v in state["weights"].values()),
        head_parameters=150 if supervised else 0,
        maximum_active_coordinates=80,
        reference_mse=root["reference_mse"],
        development=v,
        total_training_updates=13640,
        new_gradient_updates=UPDATES,
        starting_arm=root["arm"],
        starting_weights_sha=root["weights_sha"],
        starting_optimizer_sha=root["optimizer_sha"],
        starting_balance_sha=root["balance_sha"],
        warm_selected_update=root["warm_selected_update"],
        phase_batch_sha=state["sampled_batches_sha"],
        phase_noise_sha=state["sampled_noise_sha"],
    )


def run(ctx, source_path=None):
    if "test" in ctx["packs"]:
        raise ValueError("Test features must remain closed while training")
    rows = []
    for arm in ARMS:
        ctx["store"].event(
            "arm_start", arm=arm, seeds=SEEDS, new_updates_per_seed=UPDATES
        )
        for seed in SEEDS:
            root = root_for(
                ctx, seed, "vanilla5" if arm == "vanilla5_continue" else "mask5"
            )
            st = source_state(ctx, root)
            supervised = arm != "vanilla5_continue"
            both = arm == "pcgrad5_both"
            out = fit_branch(ctx, f"{arm}_s{seed}", st, root, seed, supervised, both)
            matched.complete(out, UPDATES)
            old4 = root_for(
                ctx, seed, "vanilla_continue" if not supervised else "ggf_continue"
            )
            for new, old in [
                ("sampled_batches_sha", "phase_batch_sha"),
                ("sampled_noise_sha", "phase_noise_sha"),
            ]:
                if out[new] != old4[old]:
                    raise ValueError(
                        "5D continuation differs from saved 4D image/noise stream: "
                        + new
                    )
            row = endpoint_row(ctx, seed, arm, out, root)
            rows.append(row)
            ctx["store"].write(
                f"development/{arm}_s{seed}.json", base.width.json_safe(row)
            )
            ctx["store"].save(
                "development_endpoints", dict(complete=False, models=rows)
            )
            del st, out
            gc.collect()
            torch.cuda.empty_cache()
        development_tables(ctx, rows)
        if source_path is not None:
            export(ctx, source_path)
    for seed in SEEDS:
        same = [r for r in rows if r["seed"] == seed]
        for field in ["phase_batch_sha", "phase_noise_sha"]:
            if len({r[field] for r in same}) != 1:
                raise ValueError("New 5D continuation schedules differ")
        guided = [r for r in same if r["supervised"]]
        for field in [
            "starting_weights_sha",
            "starting_optimizer_sha",
            "starting_balance_sha",
            "targets_sha",
        ]:
            if len({r[field] for r in guided}) != 1:
                raise ValueError("Guided branches do not use identical saved 5D roots")
    frozen = dict(
        complete=True,
        models=rows,
        stage=STAGE,
        test_features_opened=False,
        new_gradient_updates=27000,
    )
    ctx["store"].save("frozen_" + STAGE, frozen)
    ctx["store"].write("frozen_" + STAGE + ".json", base.width.json_safe(frozen))
    verify_freeze(ctx)
    matched.assert_sources(ctx)
    return frozen


def development_tables(ctx, rows):
    rows = list(rows) + [r for r in ctx["roots"] if r["arm"] in CONTROLS]
    values = []
    for r in rows:
        v = r["development"]
        if v.get("macro_capture") is None:
            continue
        values.append(
            dict(
                seed=r["seed"],
                arm=r["arm"],
                scheduled_updates=r.get("total_training_updates", 10640),
                capture_pct=100 * v["macro_capture"],
                patch_AP_pct=100 * v["macro_ap"],
                VE_pct=100 * v["reconstruction"]["ve"],
                MSE=v["reconstruction"]["mse"],
                native_eligible=v["reconstruction"]["mse"] <= r["reference_mse"] * 1.02,
            )
        )
    matched.csv_write(ctx["store"].results / "development_comparison.csv", values)
    means = []
    for arm in ARMS + CONTROLS:
        rr = [r for r in values if r["arm"] == arm]
        if len(rr) == 3:
            means.append(
                dict(
                    arm=arm,
                    **{
                        k: float(np.mean([r[k] for r in rr]))
                        for k in ["capture_pct", "patch_AP_pct", "VE_pct"]
                    },
                    eligible_seeds=sum(r["native_eligible"] for r in rr),
                )
            )
    matched.csv_write(ctx["store"].results / "development_means.csv", means)
    print(
        "Development means for completed three-seed arms (not test results):",
        json.dumps(means, indent=2),
        flush=True,
    )


def verify_freeze(ctx):
    frozen = memory.required(ctx["store"], "frozen_" + STAGE)
    if (
        not frozen["complete"]
        or len(frozen["models"]) != 9
        or {(r["seed"], r["arm"]) for r in frozen["models"]}
        != {(s, a) for s in SEEDS for a in ARMS}
    ):
        raise ValueError("Freeze all nine 5D continuation endpoints before test")
    for row in frozen["models"]:
        st = memory.required(ctx["store"], row["key"])
        matched.complete(st, UPDATES)
        check_state(ctx, row, st)
        root = root_for(ctx, row["seed"], row["starting_arm"])
        if any(
            row[field] != root[other]
            for field, other in [
                ("starting_weights_sha", "weights_sha"),
                ("starting_optimizer_sha", "optimizer_sha"),
                ("starting_balance_sha", "balance_sha"),
                ("targets_sha", "targets_sha"),
            ]
        ):
            raise ValueError("Frozen continuation root identity changed")
    return frozen


def evaluate(ctx):
    frozen = verify_freeze(ctx)
    matched.assert_sources(ctx)
    base.load_test(ctx)
    return evaluate_stage(ctx, STAGE, frozen)


def export(ctx, source_path):
    result = ctx["store"].load("result_" + STAGE)
    rows = ctx["store"].load("development_endpoints")
    out = dict(
        complete=result is not None and result["complete"],
        models=[] if rows is None else rows["models"],
        result=result,
        planned_new_gradient_updates=27000,
        controls_retrained=False,
        new_warmups=0,
        test_previously_inspected=True,
        new_structural_exchanges=0,
    )
    ctx["store"].write("summary.json", base.width.json_safe(out))
    if out["complete"]:
        ctx["store"].save("experiment_complete", out)
    path = matched.export(ctx, source_path)
    target = path.with_name(
        "agf_5d_continuation_" + ctx["store"].run_id[:12] + "_results.zip"
    )
    path.replace(target)
    return target


def failure(ctx, stage, exc, source_path):
    try:
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        print("Partial evidence:", export(ctx, source_path), flush=True)
    except BaseException as err:
        print("Partial export failed:", type(err).__name__, str(err), flush=True)


# BEGIN REUSED AUDIT TABLES
def evaluate_stage(ctx, stage, frozen):
    s = ctx["store"]
    reports = []
    comparisons = []
    for seed in SEEDS:
        lookup = {
            r["arm"]: audit(ctx, r) for r in frozen["models"] if r["seed"] == seed
        }
        for arm in STAGE_CONTROLS[stage]:
            row = root_for(ctx, seed, arm)
            saved = memory.required(s, f"audit_s{seed}_{arm}")
            if saved["identity"] != base.report_signature(
                row, ctx["test_stamp"], ctx["cfg"]
            ):
                raise ValueError("Saved control evaluation changed")
            base.check_report(saved["report"], row, ctx["cfg"])
            lookup[arm] = saved["report"]
        reports.extend(lookup.values())
        for a, b in PAIRS[stage]:
            ta = f"s{seed}_{a}"
            tb = f"s{seed}_{b}"
            key = f"paired_{ta}_vs_{b}"
            sig = identity(
                dict(
                    a=memory.required(s, "audit_" + ta)["identity"],
                    b=memory.required(s, "audit_" + tb)["identity"],
                    bootstrap=ctx["cfg"]["bootstrap"],
                )
            )
            old = s.load(key)
            if old is None:
                s.event(
                    "paired_intervals_start",
                    experiment=stage,
                    seed=seed,
                    treatment=a,
                    control=b,
                )
                ci = base.study.paired(
                    memory.required(s, "compact_" + ta),
                    memory.required(s, "compact_" + tb),
                    ctx["cfg"],
                )
                old = dict(identity=sig, intervals=ci)
                s.save(key, old)
            if old["identity"] != sig:
                raise ValueError("Paired interval identity changed")
            ar, br = lookup[a], lookup[b]
            comparisons.append(
                dict(
                    seed=seed,
                    treatment=a,
                    control=b,
                    paired_intervals=old["intervals"],
                    capture_gain_pp=100 * (ar["macro_capture"] - br["macro_capture"]),
                    AP_gain_pp=100 * (ar["macro_ap"] - br["macro_ap"]),
                    VE_gain_pp=100
                    * (ar["reconstruction"]["ve"] - br["reconstruction"]["ve"]),
                    reconstruction_guard_passed=ar["legacy_native_eligible"],
                    all_three_improve=all(
                        ar[k] > br[k] for k in ["macro_capture", "macro_ap"]
                    )
                    and ar["reconstruction"]["ve"] > br["reconstruction"]["ve"],
                )
            )
    out = dict(
        complete=True,
        stage=stage,
        models=reports,
        comparisons=comparisons,
        controls_retrained=False,
        new_test_gradient_updates=0,
        new_structural_exchanges=0,
        test_previously_inspected=True,
        uncertainty="Conditional whole-image paired intervals, 200 draws; no seed-level or multiplicity correction.",
    )
    stage_tables(ctx, out)
    matched.assert_sources(ctx)
    s.save("result_" + stage, out)
    s.write(stage + "_summary.json", base.width.json_safe(out))
    print(
        "\nEXPERIMENT COMPLETE:",
        stage,
        "(all three seeds). Results are saved before the next experiment.\n",
        flush=True,
    )
    return out


def stage_tables(ctx, out):
    stage = out["stage"]
    s = ctx["store"]
    rows = []
    for r in out["models"]:
        rows.append(
            dict(
                seed=r["seed"],
                arm=r["arm"],
                capture_pct=100 * r["macro_capture"],
                patch_AP_pct=100 * r["macro_ap"],
                VE_pct=100 * r["reconstruction"]["ve"],
                MSE=r["reconstruction"]["mse"],
                native_eligible=r["legacy_native_eligible"],
            )
        )
    means = []
    for arm in STAGE_ARMS[stage] + STAGE_CONTROLS[stage]:
        rr = [r for r in rows if r["arm"] == arm]
        if len(rr) != 3:
            raise ValueError("Every reported arm requires all three seeds")
        means.append(
            dict(
                arm=arm,
                **{
                    k: float(np.mean([r[k] for r in rr]))
                    for k in ["capture_pct", "patch_AP_pct", "VE_pct"]
                },
                eligible_seeds=sum(r["native_eligible"] for r in rr),
            )
        )
    matched.csv_write(s.results / (stage + "_test_comparison.csv"), rows)
    matched.csv_write(s.results / (stage + "_means.csv"), means)
    categories = [
        dict(seed=r["seed"], arm=r["arm"], **c)
        for r in out["models"]
        for c in r["categories"]
    ]
    matched.csv_write(s.results / (stage + "_categories.csv"), categories)
    flat = []
    for r in out["comparisons"]:
        ci = r["paired_intervals"]
        flat.append(
            {k: v for k, v in r.items() if k != "paired_intervals"}
            | dict(
                capture_ci_low=100 * ci["capture_gain_ci95"][0],
                capture_ci_high=100 * ci["capture_gain_ci95"][1],
                AP_ci_low=100 * ci["ap_gain_ci95"][0],
                AP_ci_high=100 * ci["ap_gain_ci95"][1],
            )
        )
    matched.csv_write(s.results / (stage + "_paired.csv"), flat)
    plot_means(s, means, stage)
    print(json.dumps(means, indent=2), flush=True)


def plot_means(store, means, stage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "guided5_continue": "5D ordinary continuation",
        "vanilla5_continue": "5D Vanilla continuation",
        "pcgrad5_both": "5D encoder + decoder PCGrad",
        "mask5": "5D starting AGF-G",
        "vanilla5": "5D starting Vanilla",
        "ggf_continue": "4D saved ordinary continuation",
        "ggf_pcgrad": "4D saved encoder PCGrad",
        "vanilla_continue": "4D saved Vanilla continuation",
        "cap040": "4D starting AGF-G",
        "vanilla4": "4D starting Vanilla",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 6.4), sharey=True, layout="constrained")
    for ax, key, title in zip(
        axes,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Capture", "Patch AP", "Reconstruction VE"],
    ):
        for j, r in enumerate(means):
            ax.barh(
                j,
                r[key],
                height=0.64,
                color="#178a7a" if r["arm"] in ARMS else "#7d90a8",
            )
            ax.text(
                max(r[key], 0) + 1,
                j,
                f"{r[key]:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
            )
        ax.set_yticks(range(len(means)), [labels[r["arm"]] for r in means], fontsize=10)
        ax.set_xlim(
            min(0, min(r[key] for r in means) - 15),
            max(100, max(r[key] for r in means) + 10),
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("%")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(
        "5D continuation / previously inspected test / three-seed means", fontsize=13
    )
    folder = store.results / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / (stage + ".png"), dpi=200)
    fig.savefig(folder / (stage + ".pdf"))
    plt.close(fig)
