"""Inference-only evaluation of completed caps 0.40 and 0.60."""

import gc, json
from pathlib import Path
import numpy as np
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import caps_pipeline as caps

base = caps.base
matched = caps.matched
frozen_weights = caps.frozen_weights
HIGHER_RUN = "08ee0d68de48ea9557a88fc1d1de24d3bf06ff7cdbb1534541758d873d595117"


def prepare(
    paths,
    training_output,
    output,
    local,
    cache_local,
    source_sha,
    previous_test_output=None,
    require_drive=True,
):
    base.safe_destination(output, list(paths.values()) + [training_output])
    ctx = base.prepare_test(
        paths, output, local, cache_local, source_sha, require_drive
    )
    source = matched.discover(
        training_output, Path(local) / "read_completed_caps", run=HIGHER_RUN
    )
    if source.config["source_sha"] != caps.LEGACY_SWEEP_SOURCE_SHA:
        raise ValueError("Unexpected higher-cap source engine")
    ctx["sources"]["sweep"] = source
    ctx["stamps"] += memory.source_snapshot([source])
    results = [memory.required(source, f"result_cap060_s{s}") for s in base.SEEDS]
    for seed in base.SEEDS:
        ctx["checkpoint_stamps"] += [
            matched.source_key_stamp(source, k)
            for k in [
                f"result_cap060_s{seed}",
                f"phase1_cap060_s{seed}",
                f"phase2_cap060_s{seed}",
            ]
        ]
    frozen = freeze_endpoints(
        ctx,
        dict(
            complete=True,
            study_complete=True,
            caps=[0.6],
            test_access=False,
            results=results,
        ),
    )
    ctx["pipeline_rows"] = frozen["models"]
    ctx["development_selection"] = frozen["development_selection"]
    ctx["store"].save("combined_models_frozen", frozen)
    ctx["store"].write(
        "completed_scope.json",
        dict(
            caps=[0.4, 0.6],
            seeds=base.SEEDS,
            models=6,
            new_gradient_updates=0,
            unfinished_caps=[0.8, 1.0],
        ),
    )
    caps.reuse_previous_cap040_audits(ctx, previous_test_output)
    ctx["store"].write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    verify_all(ctx)
    matched.assert_sources(ctx)
    return ctx


def load_test(ctx):
    verify_all(ctx)
    return base.load_test(ctx)


def export(ctx, source_path):
    original = matched.export(ctx, source_path)
    target = original.with_name(
        "ggf_completed_caps_" + ctx["store"].run_id[:12] + "_results.zip"
    )
    original.replace(target)
    return target


failure = base.failure


def freeze_endpoints(ctx, summary):
    train = ctx
    rows = []
    if (
        not summary["complete"]
        or not summary["study_complete"]
        or summary["caps"] != [0.6]
        or summary["test_access"]
    ):
        raise ValueError(
            "All three cap 0.60 development trials must complete before test evaluation"
        )
    if len(summary["results"]) != 3:
        raise ValueError("Wrong number of development endpoints")
    for seed in base.SEEDS:
        row, warm, old1, old2 = base.bound.verify_seed(train, seed)
        cap40 = next(r for r in train["cap040_rows"] if r["seed"] == seed)
        rows.append(dict(cap40, source_name="cap040", upper_bound=0.4))
        for cap in [0.6]:
            arm = f"cap{int(round(cap * 100)):03d}"
            p1 = memory.required(train["sources"]["sweep"], f"phase1_{arm}_s{seed}")
            p2 = memory.required(train["sources"]["sweep"], f"phase2_{arm}_s{seed}")
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
    selection = dict(
        selected_cap=None,
        split="development",
        test_access=False,
        evaluated_caps=[0.4, 0.6],
        policy="Evaluate the completed three-seed scope; no test selection. Caps 0.80 and 1.00 remain unfinished and are omitted.",
    )
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
        models=6,
        selected_cap=selection["selected_cap"],
        test_features_opened=False,
    )
    return frozen


def verify_all(ctx):
    frozen = memory.required(ctx["store"], "combined_models_frozen")
    expected = {
        (s, f"cap{int(round(c * 100)):03d}") for s in base.SEEDS for c in [0.4, 0.6]
    }
    if (
        not frozen["complete"]
        or len(frozen["models"]) != 6
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
            report = caps.audit_one(ctx, row)
            reports.append(report)
            lookup[row["arm"]] = report
        for arm in base.CONTROLS:
            report = memory.required(store, f"audit_s{seed}_{arm}")["report"]
            reports.append(report)
            lookup[arm] = report
        for cap in [0.4, 0.6]:
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
    arms = ["mask4_balanced", "cap040", "cap060", "vanilla4", "mask3_fixed"]
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
        "completed_caps_test",
        "Completed caps: previously inspected test, three seeds",
        labels,
    )
