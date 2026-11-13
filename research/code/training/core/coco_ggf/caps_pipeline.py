"""One uninterrupted development sweep followed by frozen test evaluation."""

import gc
import hashlib
import json
import traceback
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import cap_followup as base

matched = base.matched
LEGACY_SWEEP_SOURCE_SHA = (
    "461306be4a948a0fea8225a0c7ee80d58c78b281411a98b5e8d68fd54ef9d27a"
)
ALL_CAPS = [0.2, 0.4, 0.6, 0.8, 1.0]


def verify_unchanged_engine():
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "BASE_COMPUTE_HASHES.json").read_text())
    for name, digest in expected.items():
        if sha_file(root / name) != digest:
            raise ValueError(
                "The previously published training engine changed: " + name
            )
    return dict(
        source_sha=LEGACY_SWEEP_SOURCE_SHA,
        verified_files=len(expected),
        policy="Reuse the existing development run identity only because every numerical implementation file is byte-identical. The coordinator has its own separate source identity.",
    )


def test_input_preflight(ctx, paths):
    """Check metadata and payload hashes, without opening test features."""
    source = matched.discover(
        paths["strong"],
        ctx["local"] / "preflight_strong",
        run=ctx["parent"].config["source_runs"]["strong"],
    )
    plan = memory.required(source, "fresh_plan")
    stamp = memory.required(source, "activation_identity_test")
    ids = plan["splits"]["test"]
    if (
        len(ids) != 1500
        or len(set(ids)) != 1500
        or stamp["ids"] != ids
        or stamp["normalization_sha"] != identity(ctx["norm"])
        or stamp != ctx["parent"].config["test_identity"]
        or set(ids)
        & (
            set(ctx["provenance"]["splits"]["train"])
            | set(ctx["provenance"]["splits"]["select"])
        )
    ):
        raise ValueError("The saved test metadata differs from the control cohort")
    ctx["sources"]["strong"] = source
    ctx["stamps"] += memory.source_snapshot([source])
    chunk = source.config["config"]["extract_chunk"]
    for key in ["fresh_plan", "activation_identity_test"] + [
        f"features_test_{lo:06d}" for lo in range(0, 1500, chunk)
    ]:
        ctx["checkpoint_stamps"].append(matched.source_key_stamp(source, key))
    for seed in base.SEEDS:
        for arm in base.CONTROLS:
            row = ctx["catalog"][(seed, arm)]
            tag = f"s{seed}_{arm}"
            saved = memory.required(ctx["parent"], "audit_" + tag)
            if saved["identity"] != base.report_signature(row, stamp, ctx["cfg"]):
                raise ValueError(
                    "A saved control used different test data or scoring settings"
                )
            base.check_report(saved["report"], row, ctx["cfg"])
            ctx["checkpoint_stamps"] += [
                matched.source_key_stamp(ctx["parent"], key)
                for key in ["audit_" + tag, "compact_" + tag]
            ]
    ctx["store"].write(
        "test_input_preflight.json",
        dict(
            complete=True,
            images=1500,
            test_features_opened=False,
            controls_retrained=False,
            test_identity=stamp,
        ),
    )
    matched.assert_sources(ctx)
    return stamp


def prepare(
    paths, training_output, output, local, cache_local, source_sha, require_drive=True
):
    proof = verify_unchanged_engine()
    base.safe_destination(output, list(paths.values()) + [training_output])
    train = base.prepare_sweep(
        paths,
        training_output,
        Path(local) / "training",
        cache_local,
        LEGACY_SWEEP_SOURCE_SHA,
        require_drive,
        caps=base.CAPS,
    )
    stamp = test_input_preflight(train, paths)
    payload = dict(
        study="ggf_all_caps_train_then_test",
        version=1,
        seeds=base.SEEDS,
        new_caps=base.CAPS,
        existing_caps=[0.2, 0.4],
        source_sha=source_sha,
        training_run=train["store"].run_id,
        training_engine_source_sha=LEGACY_SWEEP_SOURCE_SHA,
        test_identity=stamp,
        maximum_new_gradient_updates=27000,
        controls_retrained=False,
        selection="Before test: maximize mean capture + existing AP weight * mean AP among caps passing the original native-MSE guard in every seed. Ties: higher VE, then smaller cap.",
        test_previously_inspected=True,
        new_structural_exchanges=0,
    )
    store = inputs.JournalStore(
        output, payload, Path(local) / "pipeline_rescue", require_drive
    )
    base.install_drive_recovery(store)
    store.write("protocol.json", payload)
    store.write("training_engine_reuse.json", proof)
    return dict(
        store=store,
        training=train,
        paths=paths,
        local=Path(local),
        cache_local=Path(cache_local),
        source_sha=source_sha,
        require_drive=require_drive,
        test_stamp=stamp,
        sources=train["sources"],
    )


def preflight(ctx):
    ctx["store"].event("training_preflight_start")
    out = base.sweep_preflight(ctx["training"])
    ctx["store"].write("preflight.json", out)
    return out


def development_decision(summary, ap_weight):
    """This rule is fixed before test; test values are not accepted as inputs."""
    candidates = {cap: [] for cap in ALL_CAPS}
    for seed in base.SEEDS:
        seed_rows = [r for r in summary["results"] if r["seed"] == seed]
        if sorted(r["upper_bound"] for r in seed_rows) != base.CAPS:
            raise ValueError(
                "Complete all three higher caps for every seed before test"
            )
        reference = seed_rows[0]["reconstruction_limit"]
        for cap, v in seed_rows[0]["baselines"].items():
            candidates[float(cap)].append(dict(seed=seed, metrics=v, limit=reference))
        for r in seed_rows:
            candidates[r["upper_bound"]].append(
                dict(
                    seed=seed, metrics=r["development"], limit=r["reconstruction_limit"]
                )
            )
    rows = []
    for cap, items in candidates.items():
        if sorted(r["seed"] for r in items) != base.SEEDS:
            raise ValueError("Development seed set changed")
        for r in items:
            base.bound.validate_metrics(r["metrics"])
        capture = float(np.mean([r["metrics"]["macro_capture"] for r in items]))
        ap = float(np.mean([r["metrics"]["macro_ap"] for r in items]))
        ve = float(np.mean([r["metrics"]["reconstruction"]["ve"] for r in items]))
        eligible = all(
            r["metrics"]["reconstruction"]["mse"] <= r["limit"] for r in items
        )
        rows.append(
            dict(
                cap=cap,
                mean_capture=capture,
                mean_AP=ap,
                mean_VE=ve,
                score=capture + ap_weight * ap,
                all_seeds_native_eligible=eligible,
            )
        )
    permitted = [r for r in rows if r["all_seeds_native_eligible"]]
    choice = (
        max(permitted, key=lambda r: (r["score"], r["mean_VE"], -r["cap"]))
        if permitted
        else None
    )
    return dict(
        selected_cap=None if choice is None else choice["cap"],
        candidates=rows,
        AP_weight=ap_weight,
        split="development",
        test_access=False,
        rule="Maximize three-seed mean capture + AP_weight * mean AP, subject to unchanged native reconstruction guard in every seed; ties: higher VE then smaller cap.",
        interpretation="A fixed development recommendation, not a claim that one cap improves every metric. All caps are still evaluated and reported on the previously inspected test.",
    )


def freeze_endpoints(ctx, summary):
    train = ctx["training"]
    rows = []
    if (
        not summary["complete"]
        or not summary["study_complete"]
        or summary["caps"] != base.CAPS
        or summary["test_access"]
    ):
        raise ValueError(
            "All nine development trials must complete before test evaluation"
        )
    if len(summary["results"]) != 9:
        raise ValueError("Wrong number of development endpoints")
    for seed in base.SEEDS:
        row, warm, old1, old2 = base.bound.verify_seed(train, seed)
        cap40 = next(r for r in train["cap040_rows"] if r["seed"] == seed)
        rows.append(dict(cap40, source_name="cap040", upper_bound=0.4))
        for cap in base.CAPS:
            arm = f"cap{int(round(cap * 100)):03d}"
            p1 = memory.required(train["store"], f"phase1_{arm}_s{seed}")
            p2 = memory.required(train["store"], f"phase2_{arm}_s{seed}")
            result = next(
                r
                for r in summary["results"]
                if r["seed"] == seed and r["upper_bound"] == cap
            )
            for phase, initial, old, n, rng, head, balance in [
                (p1, warm["best"], old1, 2500, 71000 + seed, None, None),
                (p2, p1["weights"], old2, 500, 82000 + seed, p1["head"], p1["balance"]),
            ]:
                matched.complete(phase, n)
                expected = base.phase_identity(
                    train,
                    initial,
                    row["targets"],
                    n,
                    rng,
                    row["reference_mse"],
                    base.cap_recipe(cap)["balance"],
                    head,
                    balance,
                )
                if phase["fingerprint"] != expected:
                    raise ValueError("Higher-cap phase lineage changed")
                for key in ["sampled_batches_sha", "sampled_noise_sha"]:
                    if phase[key] != old[key]:
                        raise ValueError("Higher-cap paired sampling changed")
                matched.valid_weights(phase["weights"], "vanilla", train["base"])
                if phase["history"][-1]["update"] != n:
                    raise ValueError("Incomplete final endpoint audit")
            if (
                training.weight_sha(p2["weights"]) != result["weights_sha"]
                or training.weight_sha({"head": p2["head"]}) != result["head_sha"]
                or row["targets_sha"] != result["targets_sha"]
                or training.weight_sha(warm["best"]) != result["warm_weights_sha"]
                or result["test_access"]
                or result["new_updates"] != 3000
                or result["reconstruction_limit"] != row["reference_mse"] * 1.02
                or identity(result["baselines"])
                != identity(
                    {0.2: old2["history"][-1]["validation"], 0.4: cap40["development"]}
                )
                or identity(p2["history"][-1]["validation"])
                != identity(result["development"])
            ):
                raise ValueError(
                    "The reported endpoint does not identify the saved checkpoint"
                )
            rows.append(
                dict(
                    row,
                    arm=arm,
                    label=f"Mask GGF 4: cap {cap:.2f}",
                    source_name="sweep",
                    upper_bound=cap,
                    key=f"phase2_{arm}_s{seed}",
                    origin="higher_cap_sweep",
                    weights_sha=result["weights_sha"],
                    head_sha=result["head_sha"],
                    development=result["development"],
                )
            )
            del p1, p2
            gc.collect()
        del warm, old1, old2
        gc.collect()
    selection = development_decision(summary, train["cfg"]["score_ap_weight"])
    frozen = dict(
        complete=True,
        models=rows,
        development_selection=selection,
        new_test_gradient_updates=0,
    )
    prior = ctx["store"].load("all_caps_frozen")
    if prior is not None and identity(prior) != identity(frozen):
        raise ValueError("Previously frozen endpoints or development decision changed")
    ctx["store"].save("all_caps_frozen", frozen)
    ctx["store"].write("all_caps_frozen.json", base.width.json_safe(frozen))
    ctx["store"].write("development_selection.json", selection)
    matched.assert_sources(train)
    ctx["store"].event(
        "all_caps_frozen",
        models=12,
        selected_cap=selection["selected_cap"],
        test_features_opened=False,
    )
    return frozen


def train_and_freeze(ctx):
    ctx["store"].event("development_sweep_start", new_caps=base.CAPS, seeds=base.SEEDS)
    summary = base.run_sweep(ctx["training"])
    freeze_endpoints(ctx, summary)
    return summary


def reuse_previous_cap040_audits(eval_ctx, previous_output):
    if previous_output is None:
        return []
    # The earlier evaluation notebook used this exact payload. Never choose by date.
    payload = dict(eval_ctx["store"].config, source_sha=LEGACY_SWEEP_SOURCE_SHA)
    expected = identity(payload)
    manifests = []
    for p in (Path(previous_output) / "results").glob("*/manifest.json"):
        item = json.loads(p.read_text())
        if item["run_id"] == expected:
            manifests.append(item)
    if not manifests:
        return []
    if len(manifests) != 1:
        raise ValueError("Ambiguous previous 0.40 evaluation identity")
    source = memory.FrozenStore(
        Path(previous_output) / "checkpoints" / expected[:20],
        eval_ctx["local"] / "prior_cap040",
    )
    reused = []
    for row in eval_ctx["cap040_rows"]:
        tag = f"s{row['seed']}_cap040"
        saved = source.load("audit_" + tag)
        compact = source.load("compact_" + tag)
        if saved is None and compact is None:
            continue
        if saved is None or compact is None:
            raise ValueError("Incomplete saved 0.40 audit pair")
        if saved["identity"] != base.report_signature(
            row, eval_ctx["test_stamp"], eval_ctx["cfg"]
        ):
            raise ValueError(
                "Prior 0.40 inference used a different checkpoint, readout or test"
            )
        base.check_report(saved["report"], row, eval_ctx["cfg"])
        if (
            compact["image_ids"] != eval_ctx["test_ids"]
            or compact["category_indices"] != eval_ctx["cfg"]["category_indices"]
        ):
            raise ValueError("Prior 0.40 compact cohort differs")
        eval_ctx["store"].save("audit_" + tag, saved)
        eval_ctx["store"].save("compact_" + tag, compact)
        eval_ctx["store"].write(
            "reports/" + tag + ".json", base.width.json_safe(saved["report"])
        )
        eval_ctx["checkpoint_stamps"] += [
            matched.source_key_stamp(source, key)
            for key in ["audit_" + tag, "compact_" + tag]
        ]
        reused.append(row["seed"])
        del saved, compact
        gc.collect()
    eval_ctx["sources"]["previous_cap040_evaluation"] = source
    eval_ctx["stamps"] += memory.source_snapshot([source])
    return reused


def prepare_evaluation(ctx, previous_test_output=None):
    frozen = memory.required(ctx["store"], "all_caps_frozen")
    expected = {
        (s, f"cap{int(round(c * 100)):03d}")
        for s in base.SEEDS
        for c in [0.4] + base.CAPS
    }
    if (
        not frozen["complete"]
        or len(frozen["models"]) != 12
        or {(r["seed"], r["arm"]) for r in frozen["models"]} != expected
    ):
        raise ValueError("Freeze all twelve endpoints before preparing test evaluation")
    output = ctx["store"].output / "evaluation"
    evaluation = base.prepare_test(
        ctx["paths"],
        output,
        ctx["local"] / "evaluation",
        ctx["cache_local"],
        ctx["source_sha"],
        ctx["require_drive"],
    )
    if evaluation["test_stamp"] != ctx["test_stamp"]:
        raise ValueError("Test metadata changed after preflight")
    evaluation["sources"]["sweep"] = memory.FrozenStore(
        ctx["training"]["store"].root, ctx["local"] / "read_sweep"
    )
    evaluation["stamps"] += memory.source_snapshot([evaluation["sources"]["sweep"]])
    evaluation["pipeline_rows"] = frozen["models"]
    evaluation["development_selection"] = frozen["development_selection"]
    for row in evaluation["pipeline_rows"]:
        if row["source_name"] == "sweep":
            evaluation["checkpoint_stamps"].append(
                matched.source_key_stamp(evaluation["sources"]["sweep"], row["key"])
            )
    reused = reuse_previous_cap040_audits(evaluation, previous_test_output)
    evaluation["store"].write(
        "previous_cap040_audit_reuse.json", dict(reused_seeds=reused)
    )
    evaluation["store"].save("combined_models_frozen", frozen)
    evaluation["store"].write(
        "combined_models_frozen.json", base.width.json_safe(frozen)
    )
    evaluation["store"].write(
        "combined_protocol.json",
        dict(
            study="all_caps_frozen_test",
            seeds=base.SEEDS,
            evaluated_treatment_caps=[0.4] + base.CAPS,
            controls=base.CONTROLS,
            controls_retrained=False,
            new_gradient_updates=0,
            development_selection=frozen["development_selection"],
            test_previously_inspected=True,
        ),
    )
    evaluation["store"].write(
        "source_checkpoint_hashes.json", evaluation["checkpoint_stamps"]
    )
    verify_all(evaluation)
    matched.assert_sources(evaluation)
    ctx["evaluation"] = evaluation
    ctx["store"].event(
        "test_evaluation_prepared",
        models=12,
        reused_cap040_audits=reused,
        new_gradient_updates=0,
    )
    return evaluation


def frozen_weights(ctx, row):
    state = memory.required(ctx["sources"][row["source_name"]], row["key"])
    matched.complete(state, 500)
    if (
        training.weight_sha(state["weights"]) != row["weights_sha"]
        or training.weight_sha({"head": state["head"]}) != row["head_sha"]
    ):
        raise ValueError("Frozen treatment checkpoint changed")
    return state["weights"]


def verify_all(ctx):
    frozen = memory.required(ctx["store"], "combined_models_frozen")
    expected = {
        (s, f"cap{int(round(c * 100)):03d}")
        for s in base.SEEDS
        for c in [0.4] + base.CAPS
    }
    if (
        not frozen["complete"]
        or len(frozen["models"]) != 12
        or {(r["seed"], r["arm"]) for r in frozen["models"]} != expected
        or identity(frozen["models"]) != identity(ctx["pipeline_rows"])
        or identity(frozen["development_selection"])
        != identity(ctx["development_selection"])
    ):
        raise ValueError("Frozen catalog or pre-test development decision changed")
    for row in ctx["pipeline_rows"]:
        weights = frozen_weights(ctx, row)
        matched.valid_weights(weights, "vanilla", ctx["base"])
        del weights


def load_test(ctx):
    verify_all(ctx)
    return base.load_test(ctx)


def audit_one(ctx, row):
    if row["arm"] == "cap040":
        return base.audit_treatment(ctx, row)
    store = ctx["store"]
    tag = f"s{row['seed']}_{row['arm']}"
    signature = base.report_signature(row, ctx["test_stamp"], ctx["cfg"])
    saved = store.load("audit_" + tag)
    if saved is not None:
        if saved["identity"] != signature:
            raise ValueError("Cached test audit identity changed")
        base.check_report(saved["report"], row, ctx["cfg"])
        memory.required(store, "compact_" + tag)
        return saved["report"]
    model = core.new_model(
        "vanilla", ctx["base"], frozen_weights(ctx, row), ctx["device"]
    )
    store.event("test_scoring_start", model=tag, images=1500, new_gradient_updates=0)
    with torch.inference_mode(), matched.progress(ctx, model, tag):
        report = core.validation(
            model,
            ctx["packs"]["test"],
            row["targets"],
            ctx["cfg"],
            ctx["device"],
            with_arrays=True,
        )
        compact = base.study.compact_audit(
            report.pop("_audit"), matched.mapping(row["targets"]), ctx["cfg"]
        )
        activity = matched.activity(
            model, ctx["packs"]["test"], ctx["device"], ctx["cfg"]["batch"]
        )
        dev = training.measure(
            model,
            ctx["packs"]["select"]["x"].reshape(-1, model.d),
            ctx["device"],
            ctx["cfg"]["batch"],
        )
    for key in ["mse", "ve"]:
        if abs(dev[key] - row["development"]["reconstruction"][key]) > 1e-5:
            raise ValueError(
                "Development reconstruction no longer reproduces the frozen endpoint"
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
        upper_bound=row["upper_bound"],
        activity=activity,
        development_reconstruction=dev,
        common_native_MSE_limit=row["reference_mse"] * 1.02,
        legacy_native_eligible=bool(dev["mse"] <= row["reference_mse"] * 1.02),
    )
    base.check_report(report, row, ctx["cfg"])
    store.save("compact_" + tag, compact)
    store.save("audit_" + tag, dict(identity=signature, report=report))
    store.write("reports/" + tag + ".json", base.width.json_safe(report))
    folder = store.results / "image_contrasts"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(
        folder / (tag + ".npz"),
        image_ids=np.asarray(compact["image_ids"]),
        category_indices=np.asarray(compact["category_indices"]),
        denominator=compact["denominator"],
        error=compact["error"],
    )
    store.event(
        "test_scoring_complete",
        model=tag,
        capture=report["macro_capture"],
        AP=report["macro_ap"],
        VE=report["reconstruction"]["ve"],
    )
    del model, compact
    gc.collect()
    torch.cuda.empty_cache()
    return report


def evaluate(ctx):
    verify_all(ctx)
    if "test" not in ctx["packs"]:
        raise RuntimeError("Load the verified test after freezing every endpoint")
    store = ctx["store"]
    reports = []
    comparisons = []
    for seed in base.SEEDS:
        lookup = {}
        for row in [r for r in ctx["pipeline_rows"] if r["seed"] == seed]:
            report = audit_one(ctx, row)
            reports.append(report)
            lookup[row["arm"]] = report
        for arm in base.CONTROLS:
            report = memory.required(store, f"audit_s{seed}_{arm}")["report"]
            reports.append(report)
            lookup[arm] = report
        for cap in [0.4] + base.CAPS:
            treatment = f"cap{int(round(cap * 100)):03d}"
            for control in base.CONTROLS + ([] if cap == 0.4 else ["cap040"]):
                a, b = lookup[treatment], lookup[control]
                taga = f"s{seed}_{treatment}"
                tagb = f"s{seed}_{control}"
                signature = identity(
                    dict(
                        a=memory.required(store, "audit_" + taga)["identity"],
                        b=memory.required(store, "audit_" + tagb)["identity"],
                        bootstrap=ctx["cfg"]["bootstrap"],
                    )
                )
                key = f"paired_s{seed}_{treatment}_vs_{control}"
                cached = store.load(key)
                if cached is None:
                    store.event(
                        "paired_intervals_start",
                        seed=seed,
                        treatment=treatment,
                        control=control,
                        resamples=ctx["cfg"]["bootstrap"],
                    )
                    ca = memory.required(store, "compact_" + taga)
                    cb = memory.required(store, "compact_" + tagb)
                    ci = base.study.paired(ca, cb, ctx["cfg"])
                    del ca, cb
                    gc.collect()
                    cached = dict(identity=signature, intervals=ci)
                    store.save(key, cached)
                if cached["identity"] != signature:
                    raise ValueError("Cached paired comparison changed")
                change = dict(
                    seed=seed,
                    treatment=treatment,
                    control=control,
                    capture_gain_pp=100 * (a["macro_capture"] - b["macro_capture"]),
                    AP_gain_pp=100 * (a["macro_ap"] - b["macro_ap"]),
                    VE_gain_pp=100
                    * (a["reconstruction"]["ve"] - b["reconstruction"]["ve"]),
                    paired_intervals=cached["intervals"],
                    reconstruction_guard_passed=a["legacy_native_eligible"],
                )
                comparisons.append(change)
                store.write("paired/" + key + ".json", base.width.json_safe(change))
    out = dict(
        complete=True,
        models=reports,
        comparisons=comparisons,
        new_gradient_updates=0,
        controls_retrained=False,
        new_structural_exchanges=0,
        test_previously_inspected=True,
        development_selection=ctx["development_selection"],
        uncertainty="Fixed-model paired whole-image intervals; shared dataset across seeds; no multiple-comparison correction.",
        interpretation="Report every cap, including original reconstruction-guard failures. Development recommendation was frozen before test. These are GGF models, with no structural AGF exchanges.",
    )
    tables(ctx, out)
    matched.assert_sources(ctx)
    store.save("experiment_complete", out)
    store.write("summary.json", base.width.json_safe(out))
    return out


def tables(ctx, out):
    store = ctx["store"]
    rows = []
    categories = []
    for r in out["models"]:
        rows.append(
            dict(
                seed=r["seed"],
                arm=r["arm"],
                rank=r["rank"],
                capture_pct=100 * r["macro_capture"],
                patch_AP_pct=100 * r["macro_ap"],
                VE_pct=100 * r["reconstruction"]["ve"],
                MSE=r["reconstruction"]["mse"],
                native_eligible=r["legacy_native_eligible"],
            )
        )
        categories.extend(
            dict(seed=r["seed"], arm=r["arm"], **c) for c in r["categories"]
        )
    matched.csv_write(store.results / "test_comparison.csv", rows)
    matched.csv_write(store.results / "all_categories.csv", categories)
    means = []
    arms = [
        "mask4_balanced",
        "cap040",
        "cap060",
        "cap080",
        "cap100",
        "vanilla4",
        "mask3_fixed",
    ]
    for arm in arms:
        rr = [r for r in rows if r["arm"] == arm]
        if len(rr) != 3:
            raise ValueError("Every reported arm must retain all three seeds")
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
    matched.csv_write(store.results / "three_seed_test_means.csv", means)
    paired = []
    for r in out["comparisons"]:
        ci = r["paired_intervals"]
        paired.append(
            {
                k: r[k]
                for k in [
                    "seed",
                    "treatment",
                    "control",
                    "capture_gain_pp",
                    "AP_gain_pp",
                    "VE_gain_pp",
                    "reconstruction_guard_passed",
                ]
            }
            | dict(
                capture_ci_low=100 * ci["capture_gain_ci95"][0],
                capture_ci_high=100 * ci["capture_gain_ci95"][1],
                AP_ci_low=100 * ci["ap_gain_ci95"][0],
                AP_ci_high=100 * ci["ap_gain_ci95"][1],
            )
        )
    matched.csv_write(store.results / "paired_comparison.csv", paired)
    labels = {
        f"cap{int(c * 100):03d}": f"GGF 4\nCap {c:.2f}" for c in [0.4] + base.CAPS
    }
    labels.update(
        mask4_balanced="GGF 4\nCap 0.20",
        vanilla4="Vanilla\nBSF 4",
        mask3_fixed="Original\nGGF 3",
    )
    base.plot_means(
        store,
        means,
        "all_caps_test",
        "All caps: previously inspected test, three seeds",
        labels,
    )


def complete(ctx, summary):
    frozen = memory.required(ctx["store"], "all_caps_frozen")
    if not summary["complete"] or len(summary["models"]) != 21:
        raise ValueError(
            "Complete every treatment and control audit before final packaging"
        )
    if identity(summary["development_selection"]) != identity(
        frozen["development_selection"]
    ):
        raise ValueError("The development recommendation changed during test scoring")
    out = dict(
        complete=True,
        seeds=base.SEEDS,
        new_caps=base.CAPS,
        evaluated_caps=ALL_CAPS,
        training_run=ctx["training"]["store"].run_id,
        evaluation_run=ctx["evaluation"]["store"].run_id,
        controls_retrained=False,
        development_selection=frozen["development_selection"],
        new_structural_exchanges=0,
    )
    ctx["store"].save("pipeline_complete", out)
    ctx["store"].write("summary.json", out)
    ctx["store"].event(
        "pipeline_complete",
        selected_development_cap=out["development_selection"]["selected_cap"],
    )
    return out


def export(ctx, source_path):
    store = ctx["store"]
    path = store.output / f"ggf_all_caps_{store.run_id[:12]}_results.zip"
    locations = {
        name: dict(run=s.run_id, checkpoints=str(s.root))
        for name, s in ctx["sources"].items()
    }
    locations["training"] = dict(
        run=ctx["training"]["store"].run_id,
        checkpoints=str(ctx["training"]["store"].root),
    )
    if "evaluation" in ctx:
        locations["evaluation"] = dict(
            run=ctx["evaluation"]["store"].run_id,
            checkpoints=str(ctx["evaluation"]["store"].root),
        )
    store.write("checkpoint_locations.json", locations)
    folders = dict(pipeline=store.results, development=ctx["training"]["store"].results)
    if "evaluation" in ctx:
        folders["test"] = ctx["evaluation"]["store"].results
    hashes = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for prefix, folder in folders.items():
            for p in sorted(folder.rglob("*")):
                if not p.is_file():
                    continue
                if p.suffix == ".pt":
                    raise ValueError(
                        "Large checkpoint payload cannot enter the compact evidence ZIP"
                    )
                name = prefix + "/" + p.relative_to(folder).as_posix()
                z.write(p, name)
                hashes[name] = sha_file(p)
        z.write(source_path, "source.zip")
        hashes["source.zip"] = sha_file(source_path)
        z.writestr("artifact_hashes.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise OSError("Combined evidence ZIP CRC failure")
        for name, digest in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise OSError("Combined evidence ZIP checksum failure")
    return path


def failure(ctx, stage, exc, source_path):
    try:
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                resume="Rerun the combined notebook unchanged. Completed trials and audits are reused; only unsaved training updates may replay.",
            ),
        )
        print("Partial evidence:", export(ctx, source_path), flush=True)
    except BaseException as err:
        print("Partial export failed:", type(err).__name__, str(err), flush=True)
