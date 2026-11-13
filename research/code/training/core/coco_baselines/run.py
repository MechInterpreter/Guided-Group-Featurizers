"""Bounded native BSF baseline training, frozen mappings, object-grounded audit."""

import csv
import importlib.metadata
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from . import data, metrics, training
from .storage import Store, identity, serial, sha_file

DEFAULT = dict(
    version=1,
    preset="pilot",
    images=dict(train=800, select=200, calibration=500, test=1000),
    split_seed=61001,
    patch_sample_seed=61002,
    device="cuda",
    methods=["vanilla", "grassmannian", "group_lasso"],
    groups=256,
    block_rank=3,
    active_groups=16,
    seeds=[0],
    lrs=[4e-4, 3e-3],
    epochs=40,
    batch=2048,
    snr=0.1,
    warmup=0.05,
    save_updates=200,
    local_save_updates=50,
    sparsity_tolerance=0.2,
    extract_chunk=20,
    extract_batch=8,
    audit_image_batch=8,
    detection_patches=32,
    preview_images=12,
    positive_fraction=0.5,
    negative_fraction=0.1,
    min_object_patch_mass=2.0,
    min_background_patches=4,
    min_category_images=10,
    max_audit_groups=4,
    capture_threshold=0.8,
    bootstrap=100,
)


def config(preset="pilot"):
    import copy

    cfg = copy.deepcopy(DEFAULT)
    cfg["preset"] = preset
    if preset == "smoke":
        cfg.update(
            images=dict(train=32, select=16, calibration=32, test=32),
            epochs=4,
            groups=32,
            active_groups=4,
            seeds=[0],
            lrs=[4e-4],
            batch=256,
            save_updates=30,
            local_save_updates=10,
            bootstrap=10,
            min_category_images=2,
        )
    elif preset == "confirmation":
        cfg.update(
            images=dict(train=1600, select=300, calibration=800, test=1500),
            seeds=[0, 1, 2],
            epochs=60,
            bootstrap=200,
        )
    elif preset != "pilot":
        raise ValueError("Choose smoke, pilot, or confirmation")
    return cfg


def build_store(output, local, cfg, source_sha, provenance, require_drive=True):
    versions = {
        p: importlib.metadata.version(p)
        for p in (
            "torch",
            "numpy",
            "transformers",
            "scikit-learn",
            "pycocotools",
            "einops",
        )
    }
    state = dict(
        config=cfg,
        source_sha=source_sha,
        activation_cache=provenance["cache"],
        annotation_sha=provenance["annotation_sha"],
        image_splits=provenance["splits"],
        backbone=provenance["backbone"],
        numerical_versions=versions,
    )
    # A changed environment creates a distinct identified run, never merges incompatible histories.
    s = Store(output, state, local, require_drive)
    s.write("provenance.json", provenance)
    s.write(
        "metric_definitions.json",
        dict(
            single_group_object_ap="Ability of one calibration-chosen block norm to rank object patches ahead of non-object patches. Class prevalence is the chance AP reference.",
            object_contrast_capture="1 - sum(||object-versus-rest activation contrast - native group contribution contrast||^2) / sum(||object-versus-rest activation contrast||^2). Zero means predicting no contrast. Negative values are retained.",
            contrast_spread="At the declared 80% contrast threshold, all-group and frozen multi-group capture pass but single-group capture fails. An operational contrast-distribution indicator, not an exact feature-splitting rate.",
            multi_group_probe="A fixed evaluation-only logistic probe using up to four calibration-ranked block norms. A gain indicates additional object-label evidence; it may reflect complementary attributes or probe flexibility.",
            reconstruction="Held-out variance explained by all selected group contributions.",
            uncertainty="95% exploratory image-cluster bootstrap intervals; they exclude training-seed uncertainty and multiplicity correction.",
        ),
    )
    return s


def train_baselines(store, x, selection, cfg):
    choices = []
    for method in cfg["methods"]:
        for seed in cfg["seeds"]:
            candidates = []
            for j, lr in enumerate(cfg["lrs"]):
                key = f"{method}_s{seed}_lr{j}"
                state = training.fit(
                    store, key, method, x, selection, cfg, seed, lr, cfg["device"]
                )
                candidates.append(
                    dict(
                        key=key,
                        score=state["best_score"],
                        fallback_score=state["fallback_score"],
                        best_update=state["best_update"],
                        target=state["target"],
                        parameters=state["parameters"],
                        lr=lr,
                        class_name=state["class_name"],
                        seconds=state["seconds"],
                    )
                )
            eligible = [r for r in candidates if r["score"] is not None]
            chosen = min(
                eligible or candidates,
                key=lambda r: r["score"] if eligible else r["fallback_score"],
            )
            st = store.load(chosen["key"])
            w = st["best"] if eligible else st["fallback"]
            rec = dict(
                method=method,
                seed=seed,
                **chosen,
                weights_sha=training.weight_sha(w),
                sparsity_matched=bool(eligible),
                all_candidates=candidates,
            )
            choices.append(rec)
            store.write(
                "models_frozen.json",
                dict(choices=choices, complete=False, annotations_used=False),
            )
    store.write(
        "models_frozen.json",
        dict(
            choices=choices,
            complete=True,
            annotations_used=False,
            selection="Held-out reconstruction MSE; Group Lasso target-L0 eligibility enforced.",
        ),
    )
    return choices


def _mean(rows, field):
    values = [
        r[field] for r in rows if r.get(field) is not None and np.isfinite(r[field])
    ]
    return None if not values else float(np.mean(values))


def audit_models(store, cache, norm, choices, cfg, provenance):
    completed = []
    raw_cal = data.load_split(cache, cfg, "calibration", True)
    cal_x = data.normalize(raw_cal["x"], norm)
    for choice in choices:
        key = choice["key"]
        report_key = "report_" + key
        previous = store.load(report_key)
        if previous is not None:
            completed.append(previous)
            print("Reused audit", key, flush=True)
            continue
        state = store.load(key)
        w = state["best"] if choice["sparsity_matched"] else state["fallback"]
        if training.weight_sha(w) != choice["weights_sha"]:
            raise ValueError("Frozen model changed")
        model = training.make_model(choice["method"], cfg).to(cfg["device"])
        model.load_state_dict(w)
        model.eval()
        cal = store.load("audit_cal_" + key)
        if cal is None:
            store.event("calibration_inference", key=key)
            cal = metrics.audit_arrays(
                model, cal_x, raw_cal["masks"], raw_cal["ids"], cfg, cfg["device"]
            )
            store.save("audit_cal_" + key, cal)
        frozen = store.load("mapping_" + key)
        if frozen is None:
            store.event("calibration_mapping", key=key)
            frozen = metrics.calibrate(cal, cfg, provenance["categories"])
            store.save("mapping_" + key, frozen)
            # AP matrix is retained in the checkpoint; JSON preserves interpretable choices.
            store.write(
                f"mappings/{key}.json",
                dict(
                    mappings=frozen["mappings"],
                    rule=frozen["rule"],
                    label_use=frozen["label_use"],
                    weights_sha=choice["weights_sha"],
                ),
            )
        del cal
        store.event("mapping_frozen", key=key, object_labels_used_only_for_audit=True)
        # Test annotation data are opened only after this model's mapping has frozen.
        raw_test = data.load_split(cache, cfg, "test", True)
        test_x = data.normalize(raw_test["x"], norm)
        test = store.load("audit_test_" + key)
        if test is None:
            store.event("test_inference", key=key)
            test = metrics.audit_arrays(
                model, test_x, raw_test["masks"], raw_test["ids"], cfg, cfg["device"]
            )
            store.save("audit_test_" + key, test)
        rows = metrics.evaluate(test, frozen, cfg, 61003 + choice["seed"])
        rec = training.measure(
            model, test_x.reshape(-1, 768), cfg["device"], cfg["batch"]
        )
        detectors = [r["detection"] for r in rows if r["detection"] is not None]
        contrasts = [r["contrast"] for r in rows if r["contrast"] is not None]
        latest = state["history"][-1]["selection"]["mse"]
        first = state["history"][0]["selection"]["mse"]
        summary = dict(
            method=choice["method"],
            seed=choice["seed"],
            reconstruction=rec,
            sparsity_matched=choice["sparsity_matched"],
            evaluated_detection_categories=len(detectors),
            evaluated_contrast_categories=len(contrasts),
            mean_single_group_ap=_mean(detectors, "ap"),
            mean_multi_group_probe_ap=_mean(detectors, "multi_group_probe_ap"),
            mean_single_group_contrast_capture=_mean(contrasts, "single_group_capture"),
            mean_multi_group_contrast_capture=_mean(contrasts, "multi_group_capture"),
            contrast_spread_categories=sum(
                r["distributed_object_contrast"] for r in contrasts
            ),
            learning_curve=dict(
                first_selection_mse=first,
                final_selection_mse=latest,
                best_update=state["best_update"],
                total_updates=state["updates"],
                final_mse_improvement_over_previous=(
                    state["history"][-2]["selection"]["mse"] - latest
                )
                / max(state["history"][-2]["selection"]["mse"], 1e-12)
                if len(state["history"]) > 1
                else None,
            ),
            parameters=choice["parameters"],
            category_coverage_note="Averages include only categories with sufficient visible-object images; see exclusions and counts for all 80.",
            limitation="Annotation-defined object evidence and contrast are proxies; no true latent-factor recovery or exact feature-splitting rate is available on COCO.",
        )
        result = dict(
            model=choice,
            summary=summary,
            categories=rows,
            skipped_objects=test["skipped"],
        )
        store.save(report_key, result)
        store.write(f"reports/{key}.json", result)
        completed.append(result)
        store.event("audit_complete", key=key, summaries=summary)
        del raw_test, test_x, test, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    store.write(
        "summary.json",
        dict(
            reports=[r["summary"] for r in completed],
            scope="Direct released-class COCO pilot, not a replication of published ImageNet results.",
            inference="Look for reproducible multi-group object evidence with strong reconstruction; confirm across seeds and longer training before attributing a weakness to grouping.",
        ),
    )
    export_tables(store, completed)
    return completed


def export_tables(store, results):
    rows = []
    for r in results:
        for c in r["categories"]:
            d, q = c["detection"] or {}, c["contrast"] or {}
            rows.append(
                dict(
                    method=r["model"]["method"],
                    seed=r["model"]["seed"],
                    category=c["category"],
                    calibration_positive_images=c["calibration_positive_images"],
                    test_positive_images=c["test_positive_images"],
                    single_group_ap=d.get("ap"),
                    multi_group_probe_ap=d.get("multi_group_probe_ap"),
                    single_group_f1=d.get("f1"),
                    false_positive_rate=d.get("false_positive_rate"),
                    chance_ap=d.get("prevalence"),
                    detector_group=d.get("group"),
                    contrast_group=q.get("single_group_id"),
                    same_group_selected=q.get("same_group_as_detector"),
                    detector_group_contrast_capture=q.get(
                        "detector_group_contrast_capture"
                    ),
                    contrast_group_object_ap=q.get("chosen_group_object_ap"),
                    single_group_contrast_capture=q.get("single_group_capture"),
                    multi_group_contrast_capture=q.get("multi_group_capture"),
                    all_group_contrast_capture=q.get("all_group_capture"),
                    contrast_group_count=q.get("k_for_threshold"),
                    distributed_object_contrast=q.get("distributed_object_contrast"),
                    exclusions="; ".join(c["exclusions"]),
                )
            )
    if rows:
        p = store.rescue / "per_category.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        store.check_mount()
        shutil.copyfile(p, store.results / "per_category.csv")


def plots(store, results):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figdir = store.rescue / "figures"
    figdir.mkdir(exist_ok=True)
    colors = dict(vanilla="#008d83", grassmannian="#8462a9", group_lasso="#d18826")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for r in results:
        label = f"{r['model']['method']} seed {r['model']['seed']}"
        color = colors[r["model"]["method"]]
        ds = [c["detection"] for c in r["categories"] if c["detection"] is not None]
        qs = [c["contrast"] for c in r["categories"] if c["contrast"] is not None]
        axes[0].scatter(
            [d["ap"] for d in ds],
            [d["multi_group_probe_ap"] for d in ds],
            color=color,
            s=25,
            alpha=0.55,
            label=label,
        )
        axes[1].scatter(
            [q["single_group_capture"] for q in qs],
            [q["multi_group_capture"] for q in qs],
            color=color,
            s=25,
            alpha=0.55,
        )
        hist = store.load(r["model"]["key"])["history"]
        axes[2].plot(
            [h["epoch"] for h in hist],
            [h["selection"]["ve"] for h in hist],
            color=color,
            label=label,
        )
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(
        xlabel="Single-group object AP",
        ylabel="Up-to-four-group probe AP",
        title="Extra object evidence across groups",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    limits = axes[1].get_xlim()
    low = min(-0.2, limits[0])
    axes[1].plot([low, 1], [low, 1], "--", color="gray")
    axes[1].set(
        xlabel="Single-group object-contrast capture",
        ylabel="Frozen multi-group contrast capture",
        title="Object contrast concentration",
    )
    axes[2].set(
        xlabel="Training epochs",
        ylabel="Selection reconstruction VE",
        title="Check training maturity",
    )
    axes[0].legend(fontsize=8)
    [ax.grid(alpha=0.15) for ax in axes]
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"baseline_overview.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    for r in results:
        rows = [c for c in r["categories"] if c["contrast"] is not None]
        # All eligible categories, ordered by name; never hide inconvenient categories.
        rows = sorted(rows, key=lambda c: c["category"])
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.35), 5))
        at = np.arange(len(rows))
        q = [c["contrast"] for c in rows]
        ax.bar(
            at - 0.18,
            [c["single_group_capture"] for c in q],
            width=0.36,
            label="One frozen group",
            color="#008d83",
        )
        ax.bar(
            at + 0.18,
            [c["multi_group_capture"] for c in q],
            width=0.36,
            label="Up to four frozen groups",
            color="#9a8cad",
        )
        ax.axhline(0.8, ls="--", color="gray", label="Declared contrast threshold")
        ax.set_xticks(at, [c["category"] for c in rows], rotation=70, ha="right")
        ax.set(
            ylabel="Object-contrast explained energy",
            title=r["model"]["key"]
            + " · annotation-defined contrast, not latent-factor recovery",
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            figdir / (r["model"]["key"] + "_contrast.png"), dpi=160, bbox_inches="tight"
        )
        plt.close(fig)
    store.check_mount()
    (store.results / "figures").mkdir(exist_ok=True)
    for p in figdir.iterdir():
        shutil.copyfile(p, store.results / "figures" / p.name)


def export(store, source):
    store.check_mount()
    dest = store.output / f"agf_coco_bsf_baseline_{store.run_id[:12]}_results.zip"
    local = store.rescue / "results.zip"
    with zipfile.ZipFile(local, "w", zipfile.ZIP_DEFLATED) as z:
        for p in store.results.rglob("*"):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(store.results).as_posix())
        for p in Path(source).rglob("*"):
            if p.is_file() and ("__pycache__" not in p.parts):
                z.write(p, "source/" + p.relative_to(source).as_posix())
    shutil.copyfile(local, dest)
    store.write(
        "export.json",
        dict(
            results_zip=str(dest),
            sha256=sha_file(dest),
            checkpoints=str(store.root),
            local_rescue=str(store.rescue),
            note="All raw activations, aligned masks, norm statistics, model/optimizer states and detailed audit arrays stay on Drive. The ZIP contains reports, plots, histories, frozen mappings and exact source.",
        ),
    )
    return dest
