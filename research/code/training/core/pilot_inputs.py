"""Evaluate exactly six saved pilot endpoints. No fitting or feature extraction."""

import argparse, copy, gc, json, os, subprocess, sys, time, traceback, zipfile
from pathlib import Path
import numpy as np
import torch
from scipy.stats import t as student_t
from coco_baselines import data, training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import inputs
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched, study
from coco_ggf import twelve_capacity as source_protocol
from scale_metrics import evaluate
from scale_models import model

CORE_SHA = "8a29854a50c94420686f9b904f7eaecd88f9324bcf2b016c1ce6ad4057aa0b9c"
PILOT_RUN = "bd3eb87f9fb5ffc0a41963b4e8154c881e32b96cb44b8c40e8730665a72fff32"
SEEDS = [21, 22, 23]
KINDS = ["agf_g", "released_vanilla"]


def catalog(summary):
    rows = summary.get("models", [])
    if (
        not summary.get("complete")
        or summary.get("final_test_opened")
        or summary.get("seven_seed_started")
    ):
        raise ValueError("Require the completed development-only pilot")
    if len(rows) != 6 or {(r["seed"], r["kind"]) for r in rows} != {
        (s, k) for s in SEEDS for k in KINDS
    }:
        raise ValueError("Exactly six pilot endpoints are required")
    for r in rows:
        if (
            r["groups"],
            r["rank"],
            r["active_groups"],
            r["additional_updates"],
            r["total_training_updates"],
        ) != (4096, 12, 32, 3000, 10640):
            raise ValueError("Wrong pilot configuration or budget")
        if (
            r["method"] != "released_vanilla"
            or r["coefficient"] != 0.001
            or r["lr"] != 0.0003
        ):
            raise ValueError("Wrong pilot implementation or training recipe")
        if r["cap"] != (0.6 if r["kind"] == "agf_g" else None):
            raise ValueError("Wrong pilot adaptive cap")
        if identity(r["targets"]) != r["targets_sha"]:
            raise ValueError("TRAIN readout hash changed")
    return [
        copy.deepcopy(next(r for r in rows if (r["seed"], r["kind"]) == (s, k)))
        for s in SEEDS
        for k in KINDS
    ]


def validate_targets(row, categories):
    if [t["category_index"] for t in row["targets"]] != categories:
        raise ValueError("Fixed category cohort changed")
    for t in row["targets"]:
        for key in ["contrast_group", "detector_group"]:
            if not isinstance(t[key], int) or not 0 <= t[key] < row["groups"]:
                raise ValueError("Invalid TRAIN-selected readout")
    if identity(row["targets"]) != row["targets_sha"]:
        raise ValueError("Readout identity changed")


def validate_state(row, state):
    if (
        state["updates"] != 3000
        or state["target"] != 12000
        or training.weight_sha(state["weights"]) != row["weights_sha"]
    ):
        raise ValueError(
            "Pilot endpoint changed or was continued; no retraining fallback"
        )
    if row["kind"] == "agf_g":
        if (
            state["head"] is None
            or tuple(state["head"].shape) != (len(row["targets"]), 2)
            or not torch.isfinite(state["head"]).all()
        ):
            raise ValueError("Missing trained pilot heads")
    elif state["head"] is not None or state["balance"] is not None:
        raise ValueError("Vanilla must be reconstruction-only")


def prepare(options, require_drive=True):
    local = Path(options["local"])
    local.mkdir(parents=True, exist_ok=True)
    source_protocol.cache.safe_destination(
        options["output"], [options["pilot_output"], *options["paths"].values()]
    )
    pilot = matched.discover(
        options["pilot_output"], local / "read_pilot", run=PILOT_RUN
    )
    if (
        pilot.config["source_sha"] != CORE_SHA
        or pilot.config["versions"] != matched.versions()
    ):
        raise ValueError("Pilot source/numerical versions changed")
    summary_path = pilot.results / "priority_pilot_summary.json"
    rows = catalog(json.loads(summary_path.read_text()))
    capacity = matched.discover(
        options["paths"]["capacity"],
        local / "read_capacity",
        run=source_protocol.PARENT_RUN,
    )
    if (
        capacity.config["source_sha"] != source_protocol.PARENT_SOURCE_SHA
        or capacity.config["versions"] != matched.versions()
    ):
        raise ValueError("Cached activation lineage changed")
    original = matched.discover(
        options["paths"]["original"],
        local / "read_original",
        run=pilot.config["activation_run"],
    )
    strong = matched.discover(
        options["paths"]["strong"],
        local / "read_strong",
        run=capacity.config["strong_run"],
    )
    if original.run_id != memory.EXPECTED_PARENT:
        raise ValueError("Wrong original activation cache")
    norm = memory.required(original, "normalization")
    provenance = json.loads((original.results / "provenance.json").read_text())
    plan = memory.required(strong, "fresh_plan")
    stamp = memory.required(strong, "activation_identity_test")
    ids = plan["splits"]["test"]
    if (
        len(ids) != 1500
        or len(set(ids)) != 1500
        or stamp["ids"] != ids
        or stamp != capacity.config["test_identity"]
        or stamp["normalization_sha"] != identity(norm)
        or pilot.config["normalization_sha"] != identity(norm)
        or set(ids)
        & set(
            i for s in ["train", "select", "confirm"] for i in provenance["splits"][s]
        )
    ):
        raise ValueError("Wrong cached test cohort, normalization or split overlap")
    if (
        np.asarray(norm["mean"]).shape != (196, 768)
        or not np.isfinite(norm["mean"]).all()
        or not np.isfinite(norm["scale"])
        or norm["scale"] <= 0
    ):
        raise ValueError("Invalid saved normalization")
    cfg = dict(
        capacity.config["fit_config"],
        batch=512,
        audit_image_batch=1,
        bootstrap=1000,
        fresh_test_seed=126101,
        preview_images=0,
    )
    if len(cfg["category_indices"]) != 75:
        raise ValueError("Expected all 75 categories")
    for row in rows:
        validate_targets(row, cfg["category_indices"])
    payload = dict(
        study="completed_4096_pilot_test_only",
        source_sha=options["wrapper_sha"],
        core_sha=CORE_SHA,
        pilot_run=PILOT_RUN,
        versions=matched.versions(),
        cfg=cfg,
        test_identity=stamp,
        normalization_sha=identity(norm),
        models=rows,
        test_previously_inspected=True,
        new_gradient_updates=0,
        new_feature_extractions=0,
        seven_seed_study=False,
        exchanges=False,
    )
    store = inputs.JournalStore(
        options["output"], payload, local / "rescue", require_drive
    )
    ctx = dict(
        store=store,
        parent=pilot,
        rows=rows,
        local=local,
        device="cuda",
        cfg=cfg,
        norm=norm,
        packs={},
        test_ids=ids,
        test_stamp=stamp,
        sources=dict(strong=strong, original=original),
        test_chunk=strong.config["config"]["extract_chunk"],
        stamps=memory.source_snapshot([pilot, capacity, original, strong]),
        checkpoint_stamps=[],
    )
    ctx["stamps"] += [
        dict(path=str(p), sha256=sha_file(p))
        for p in [summary_path, original.results / "provenance.json"]
    ]
    for src, keys in [
        (original, ["normalization"]),
        (strong, ["fresh_plan", "activation_identity_test"]),
    ]:
        ctx["checkpoint_stamps"] += [matched.source_key_stamp(src, k) for k in keys]
    # Validate all six endpoints and preserve only lightweight local weight caches.
    # The source is a FrozenStore: all parent write operations raise.
    weights_dir = local / "verified_weights"
    weights_dir.mkdir(exist_ok=True)
    streams = {}
    for row in rows:
        tag = f"s{row['seed']}_{row['kind']}"
        path = weights_dir / (tag + ".pt")
        meta = path.with_suffix(".json")
        cached = path.exists() and meta.exists()
        if cached:
            record = json.loads(meta.read_text())
            cached = record["weights_sha"] == row["weights_sha"] and record[
                "file_sha"
            ] == sha_file(path)
        if cached:
            memory.assert_immutable([record["source_meta"]])
            pair = record["streams"]
        else:
            state = memory.required(pilot, row["key"])
            validate_state(row, state)
            pair = [state["sampled_batches_sha"], state["sampled_noise_sha"]]
            source_stamp = matched.source_key_stamp(pilot, row["key"])
            source_meta = dict(
                path=source_stamp["meta_path"], sha256=source_stamp["meta_sha"]
            )
            pending = path.with_suffix(".pending.pt")
            torch.save(state["weights"], pending)
            os.replace(pending, path)
            record = dict(
                weights_sha=row["weights_sha"],
                file_sha=sha_file(path),
                source_meta=source_meta,
                streams=pair,
            )
            meta.write_text(json.dumps(record, indent=2))
            del state
            gc.collect()
        if row["seed"] in streams and streams[row["seed"]] != pair:
            raise ValueError("Pilot training image/noise streams differ within a seed")
        streams[row["seed"]] = pair
        ctx["stamps"].append(record["source_meta"])
        store.event(
            "pilot_endpoint_verified", seed=row["seed"], kind=row["kind"], updates=10640
        )
    frozen = dict(
        complete=True,
        models=rows,
        all_six_frozen_before_test=True,
        new_gradient_updates=0,
        test_previously_inspected=True,
    )
    old = store.load("pilot_models_frozen")
    if old is not None and identity(old) != identity(frozen):
        raise ValueError("Frozen pilot catalog changed")
    store.save("pilot_models_frozen", frozen)
    store.write("pilot_models_frozen.json", frozen)
    store.write("protocol.json", payload)
    store.write("test_identity.json", stamp)
    matched.assert_sources(ctx)
    return ctx


def verify_frozen(ctx):
    frozen = memory.required(ctx["store"], "pilot_models_frozen")
    if not frozen.get("complete") or identity(frozen["models"]) != identity(
        ctx["rows"]
    ):
        raise ValueError("Freeze exactly the declared six endpoints first")
    if len(ctx["rows"]) != 6 or {(r["seed"], r["kind"]) for r in ctx["rows"]} != {
        (s, k) for s in SEEDS for k in KINDS
    }:
        raise ValueError("Incomplete six-model pilot")
    matched.assert_sources(ctx)
    return frozen


def load_cached_test(ctx):
    verify_frozen(ctx)
    source = ctx["sources"]["strong"]
    ids = ctx["test_ids"]
    stamp = ctx["test_stamp"]
    norm = ctx["norm"]
    n = len(ids)
    folder = ctx["local"] / "normalized"
    folder.mkdir(exist_ok=True)
    path = folder / ("pilot_test_" + stamp["normalized_sha"] + ".npy")
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
            shape=(n, 196, np.asarray(norm["mean"]).shape[-1]),
        )
    masks = []
    for lo in range(0, n, ctx["test_chunk"]):
        shard = memory.required(source, f"features_test_{lo:06d}")
        want = ids[lo : lo + ctx["test_chunk"]]
        if (
            shard["ids"] != want
            or shard["normalization_sha"] != identity(norm)
            or len(shard["masks"]) != len(want)
        ):
            raise ValueError("Cached test shard identity changed")
        raw = (
            shard["x"].float().numpy()
            if torch.is_tensor(shard["x"])
            else np.asarray(shard["x"], np.float32)
        )
        if (
            raw.shape != (len(want), 196, np.asarray(norm["mean"]).shape[-1])
            or not np.isfinite(raw).all()
        ):
            raise ValueError("Invalid test features")
        if target is not None:
            target[lo : lo + len(want)] = data.normalize(raw, norm)
        masks.extend(shard["masks"])
        del shard, raw
        if lo % 200 == 0:
            ctx["store"].event(
                "loading_cached_test", completed=min(lo + ctx["test_chunk"], n), total=n
            )
    if target is not None:
        target.flush()
        del target
        os.replace(path.with_suffix(".pending.npy"), path)
    x = np.load(path, mmap_mode="r")
    if (
        x.shape != (n, 196, np.asarray(norm["mean"]).shape[-1])
        or x.dtype != np.float32
        or memory.streaming_array_sha(x) != stamp["normalized_sha"]
    ):
        raise ValueError("Cached test activation bytes differ")
    meta.write_text(
        json.dumps(
            dict(file_sha=sha_file(path), normalized_sha=stamp["normalized_sha"])
        )
    )
    pack = dict(x=x, ids=ids, masks=masks)
    ctx["packs"]["test"] = pack
    matched.assert_sources(ctx)
    return pack


def seed_stats(values):
    a = np.asarray(values, float)
    if len(a) != 3 or not np.isfinite(a).all():
        raise ValueError("Exactly three finite seeds required")
    mean = float(a.mean())
    sd = float(a.std(ddof=1))
    half = float(student_t.ppf(0.975, 2) * sd / np.sqrt(3))
    return dict(mean=mean, seed_sd=sd, seed_ci95=[mean - half, mean + half])


def tables(ctx, reports):
    rows = [
        dict(
            seed=r["seed"],
            kind=r["kind"],
            label="AGF" if r["kind"] == "agf_g" else "Vanilla BSF",
            capture_pct=100 * r["macro_capture"],
            patch_AP_pct=100 * r["macro_ap"],
            VE_pct=100 * r["reconstruction"]["ve"],
            active_groups=r["reconstruction"]["active_groups"],
            updates=r["total_training_updates"],
            native_warm_eligible=r["native_warm_eligible"],
        )
        for r in reports
    ]
    matched.csv_write(ctx["store"].results / "test_models.csv", rows)
    means = []
    for kind in KINDS:
        rr = [r for r in rows if r["kind"] == kind]
        if len(rr) != 3:
            continue
        entry = dict(
            kind=kind, label="AGF" if kind == "agf_g" else "Vanilla BSF", seeds=3
        )
        for k in ["capture_pct", "patch_AP_pct", "VE_pct"]:
            stat = seed_stats([r[k] for r in rr])
            entry.update(
                {
                    k: stat["mean"],
                    k + "_seed_sd": stat["seed_sd"],
                    k + "_seed_ci_low": stat["seed_ci95"][0],
                    k + "_seed_ci_high": stat["seed_ci95"][1],
                }
            )
        means.append(entry)
    if means:
        matched.csv_write(ctx["store"].results / "test_means.csv", means)
    return rows, means


def score_models(ctx, pack):
    frozen = verify_frozen(ctx)
    s = ctx["store"]
    reports = []
    cohort = identity(
        dict(
            test=ctx["test_stamp"], cfg=ctx["cfg"], normalization=identity(ctx["norm"])
        )
    )
    if pack["ids"] != ctx["test_ids"]:
        raise ValueError("Wrong test image order")
    for row in frozen["models"]:
        tag = f"s{row['seed']}_{row['kind']}"
        key = "audit_" + tag
        signature = identity(
            dict(weights=row["weights_sha"], targets=row["targets_sha"], cohort=cohort)
        )
        saved = s.load(key)
        if saved is None:
            started = time.monotonic()
            s.event("test_scoring_start", model=tag, images=len(pack["ids"]))
            weights = torch.load(
                ctx["local"] / "verified_weights" / (tag + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
            if training.weight_sha(weights) != row["weights_sha"]:
                raise ValueError("Local frozen weights changed")
            m = model(
                row["method"],
                row["groups"],
                row["rank"],
                row["active_groups"],
                pack["x"].shape[-1],
                row["coefficient"],
            ).to(ctx["device"])
            m.load_state_dict(weights)
            del weights
            m.eval()
            with matched.progress(ctx, m, tag):
                report = evaluate(
                    m, pack, row["targets"], ctx["cfg"], ctx["device"], compact=True
                )
            compact = report.pop("_compact")
            native = next(
                r["reference_warm_mse"]
                for r in frozen["models"]
                if r["seed"] == row["seed"] and r["kind"] == "released_vanilla"
            )
            report.update(
                seed=row["seed"],
                kind=row["kind"],
                weights_sha=row["weights_sha"],
                targets_sha=row["targets_sha"],
                groups=row["groups"],
                rank=row["rank"],
                active_groups=row["active_groups"],
                total_training_updates=row["total_training_updates"],
                native_warm_eligible=row["development"]["reconstruction"]["mse"]
                <= native * 1.02,
                development=row["development"],
                elapsed_seconds=time.monotonic() - started,
            )
            saved = dict(signature=signature, report=report, compact=compact)
            s.save(key, saved)
            s.write("reports/" + tag + ".json", report)
            del m, compact
            gc.collect()
            if ctx["device"].startswith("cuda"):
                torch.cuda.empty_cache()
        if saved["signature"] != signature:
            raise ValueError("Saved evaluation identity differs")
        compact = saved["compact"]
        if (
            compact["image_ids"] != pack["ids"]
            or compact["category_indices"] != ctx["cfg"]["category_indices"]
        ):
            raise ValueError("Paired audit cohort changed")
        reports.append(saved["report"])
        partial_rows, _ = tables(ctx, reports)
        print("TEST RESULT", json.dumps(partial_rows[-1]), flush=True)
        del compact, saved
        gc.collect()
    rows, means = tables(ctx, reports)
    out = dict(
        complete=True,
        cohort_sha=cohort,
        models=reports,
        means=means,
        new_gradient_updates=0,
        test_previously_inspected=True,
        images=len(pack["ids"]),
    )
    s.save("point_metrics_complete", out)
    s.write("test_point_summary.json", out)
    print(
        "ALL SIX TEST POINT ESTIMATES SAVED. Three-seed means:",
        json.dumps(means),
        flush=True,
    )
    plot(s.results / "figures", means)
    return out


def paired_results(ctx, points):
    s = ctx["store"]
    comparisons = []
    for seed in SEEDS:
        a = memory.required(s, f"audit_s{seed}_agf_g")
        b = memory.required(s, f"audit_s{seed}_released_vanilla")
        signature = identity(
            dict(
                treatment=a["signature"],
                control=b["signature"],
                bootstrap=ctx["cfg"]["bootstrap"],
                seed=ctx["cfg"]["fresh_test_seed"],
            )
        )
        key = f"paired_s{seed}"
        saved = s.load(key)
        if saved is None:
            s.event(
                "paired_intervals_start", seed=seed, replicates=ctx["cfg"]["bootstrap"]
            )
            ci = study.paired(a["compact"], b["compact"], ctx["cfg"])
            saved = dict(signature=signature, intervals=ci)
            s.save(key, saved)
        if saved["signature"] != signature:
            raise ValueError("Paired-interval identity changed")
        ar, br = a["report"], b["report"]
        ci = saved["intervals"]
        comparisons.append(
            dict(
                seed=seed,
                capture_gain_pp=100 * (ar["macro_capture"] - br["macro_capture"]),
                AP_gain_pp=100 * (ar["macro_ap"] - br["macro_ap"]),
                VE_gain_pp=100
                * (ar["reconstruction"]["ve"] - br["reconstruction"]["ve"]),
                capture_ci_low=100 * ci["capture_gain_ci95"][0],
                capture_ci_high=100 * ci["capture_gain_ci95"][1],
                AP_ci_low=100 * ci["ap_gain_ci95"][0],
                AP_ci_high=100 * ci["ap_gain_ci95"][1],
                valid_bootstraps=ci["valid_bootstrap_replicates"],
                invalid_bootstraps=ci["invalid_full_cohort_replicates"],
            )
        )
        s.write(f"paired/seed{seed}.json", saved)
        matched.csv_write(s.results / "test_paired_images.csv", comparisons)
        del a, b, saved
        gc.collect()
    aggregate = {}
    for metric in ["capture_gain_pp", "AP_gain_pp", "VE_gain_pp"]:
        stat = seed_stats([r[metric] for r in comparisons])
        aggregate.update(
            {
                metric: stat["mean"],
                metric + "_seed_sd": stat["seed_sd"],
                metric + "_seed_ci_low": stat["seed_ci95"][0],
                metric + "_seed_ci_high": stat["seed_ci95"][1],
            }
        )
    matched.csv_write(s.results / "test_paired_seeds.csv", [aggregate])
    matched.csv_write(
        s.results / "test_categories.csv",
        [
            dict(seed=r["seed"], kind=r["kind"], **cat)
            for r in points["models"]
            for cat in r["categories"]
        ],
    )
    out = dict(
        points,
        paired_comparisons=comparisons,
        paired_seed_summary=aggregate,
        interpretation="Three fixed pilot seeds on the previously inspected coverage-enriched test cohort. Image intervals and seed variation are distinct; no multiplicity adjustment. No configuration selection from these test scores.",
        seven_seed_study=False,
        fresh_holdout_opened=False,
        new_gradient_updates=0,
        exchanges=False,
    )
    matched.assert_sources(ctx)
    s.save("test_complete", out)
    s.write("test_summary.json", out)
    return out


def plot(folder, means):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), layout="constrained")
    for ax, key, title in zip(
        axes,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Capture", "Patch AP", "Reconstruction VE"],
    ):
        bars = ax.barh(
            range(2),
            [r[key] for r in means],
            color=["#087f83", "#879cad"],
            height=0.55,
            xerr=[r[key + "_seed_sd"] for r in means],
            capsize=3,
        )
        ax.bar_label(
            bars, labels=[f"{r[key]:.2f}%" for r in means], padding=7, fontsize=11
        )
        ax.set(
            yticks=range(2),
            yticklabels=[r["label"] for r in means],
            xlim=(0, max(105, max(r[key] for r in means) + 18)),
            xlabel="Percent; error bars = seed SD",
            title=title,
        )
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "4,096 groups / 12D / Top-32 / three seeds / 10,640 updates\nPreviously inspected test cohort; AGF cap 0.60"
    )
    for ext in ["pdf", "png"]:
        fig.savefig(folder / ("pilot_test_comparison." + ext), dpi=200)
    plt.close(fig)


def export(ctx, options, complete):
    s = ctx["store"]
    s.write(
        "completion_status.json",
        dict(
            complete=complete,
            point_metrics_complete=s.load("point_metrics_complete") is not None,
            paired_intervals_complete=s.load("test_complete") is not None,
            new_gradient_updates=0,
            new_feature_extractions=0,
            seven_seed_study=False,
            fresh_holdout_opened=False,
            exchanges=False,
        ),
    )
    target = s.output / f"agf_pilot_4096_test_{s.run_id[:12]}_results.zip"
    hashes = {}
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(s.results.rglob("*")):
            if p.is_file():
                name = "results/" + p.relative_to(s.results).as_posix()
                z.write(p, name)
                hashes[name] = sha_file(p)
        for n, k in [
            ("core_source.zip", "source_zip"),
            ("pilot_test_source.zip", "wrapper_zip"),
        ]:
            p = Path(options[k])
            z.write(p, n)
            hashes[n] = sha_file(p)
        z.writestr("artifact_hashes.json", json.dumps(hashes, indent=2))
    import hashlib

    with zipfile.ZipFile(target) as z:
        if z.testzip() is not None or any(
            hashlib.sha256(z.read(n)).hexdigest() != h for n, h in hashes.items()
        ):
            raise ValueError("Evidence ZIP checksum failed")
    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    options = json.loads(Path(p.parse_args().config).read_text())
    if options["source_sha"] != CORE_SHA or options["new_training"] or options["fresh"]:
        raise ValueError("This notebook is saved-pilot evaluation only")
    if matched.versions() != options["expected_versions"]:
        raise ValueError("Numerical packages differ from saved training")
    wrapper = Path(__file__).resolve().parent
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(wrapper / "test_pilot_test.py"),
            "--disable-warnings",
            "--basetemp",
            str(Path(options["local"]) / ("cpu_checks_" + os.urandom(4).hex())),
        ],
        cwd=wrapper,
        env=env,
    )
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("Select a T4 GPU")
    from threadpoolctl import threadpool_limits

    threadpool_limits(2)
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True, warn_only=True)
    ctx = None
    stage = "verify_saved_pilot"
    try:
        ctx = prepare(options)
        ctx["store"].results.joinpath("PROTOCOL.md").write_bytes(
            (wrapper / "PROTOCOL.md").read_bytes()
        )
        stage = "load_existing_test_cache"
        pack = load_cached_test(ctx)
        stage = "six_saved_model_test_evaluations"
        points = score_models(ctx, pack)
        export(ctx, options, False)
        del pack
        ctx["packs"].clear()
        gc.collect()
        stage = "three_paired_comparisons"
        result = paired_results(ctx, points)
        archive = export(ctx, options, True)
        Path(options["receipt"]).write_text(
            json.dumps(
                dict(
                    complete=True,
                    result_zip=str(archive),
                    results=str(ctx["store"].results),
                    wrapper_sha=options["wrapper_sha"],
                    models=6,
                    seeds=SEEDS,
                    new_training_updates=0,
                    fresh_holdout_opened=False,
                    seven_seed_study=False,
                ),
                indent=2,
            )
        )
        print(
            "PILOT TEST COMPLETE. STOPPED FOR REVIEW.",
            json.dumps(result["means"]),
            archive,
            flush=True,
        )
    except BaseException as exc:
        if ctx is not None:
            try:
                ctx["store"].write(
                    "interruption.json",
                    dict(
                        stage=stage,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    ),
                )
                print("Partial results:", export(ctx, options, False), flush=True)
            except BaseException as secondary:
                print("Partial export failed:", repr(secondary), flush=True)
        raise


if __name__ == "__main__":
    main()
