"""One-seed reconstruction-only Vanilla width probe, with immutable parent caches."""

import copy
import gc
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from coco_baselines import metrics, training
from coco_baselines.storage import identity, sha_file
from coco_adaptation import inputs
from coco_coordinated import run as memory
from coco_ggf import study

EXPECTED = "95e7186b193ed3744ced01585a0ff0f4a198c7318a53ab23cde20b4adb5082fd"
EXPECTED_SOURCE = "eb291dc32c7989cd3f64b5aa4cf46c8369ab6c7fd50b764f7f2f8a326371f7df"
NUMERIC = ("torch", "numpy", "einops")


def settings(device="cuda"):
    return dict(
        version=1,
        study="vanilla_bsf_width_probe",
        seed=21,
        widths=[1024, 4096],
        device=device,
        bootstrap=100,
        group_chunk=64,
        metric_seed=71004,
        evaluation_scope="Historical confirmation development split; one-seed capacity diagnostic",
    )


def reuse_policy(saved, current):
    differences = {
        k: dict(saved=saved.get(k), current=current.get(k))
        for k in NUMERIC
        if saved.get(k) != current.get(k)
    }
    return dict(
        reuse_1024=not differences,
        differences=differences,
        decision="Reuse verified native 1024"
        if not differences
        else "Train both widths under the current numerical environment",
    )


def coverage_check(pack, cfg, categories):
    """Validate the exact detection sample and contrast eligibility before spending GPU time."""
    p = min(cfg["detection_patches"], 196)
    pos = []
    neg = []
    objects = {c: set() for c in cfg["category_indices"]}
    for i, mm in enumerate(pack["masks"]):
        pick = np.random.default_rng(
            cfg["patch_sample_seed"] + int(pack["ids"][i])
        ).choice(196, p, replace=False)
        y, v = metrics.binary_labels(
            dict(fraction=mm["fraction"][pick], crowd=mm["crowd"][pick]), cfg
        )
        pos.append((y & v).any(0))
        neg.append((~y & v).any(0))
        known = mm["crowd"] <= cfg["negative_fraction"]
        for obj in mm["instances"]:
            c = obj["category"]
            if c not in objects:
                continue
            fg = np.asarray(obj["fraction"], np.float64) * known
            bg = (
                mm["fraction"][:, c].astype(float) / 255 <= cfg["negative_fraction"]
            ) & known
            if (
                fg.sum() >= cfg["min_object_patch_mass"]
                and bg.sum() >= cfg["min_background_patches"]
            ):
                objects[c].add(i)
    ps = np.asarray(pos).sum(0)
    ns = np.asarray(neg).sum(0)
    return [
        dict(
            category_index=c,
            category=categories[c]["name"],
            positive_images=int(ps[c]),
            negative_images=int(ns[c]),
            contrast_images=len(objects[c]),
            supported=bool(
                min(ps[c], ns[c], len(objects[c])) >= cfg["min_category_images"]
            ),
        )
        for c in cfg["category_indices"]
    ]


def prepare(
    source_output,
    activation_output,
    output,
    local,
    source_sha,
    versions,
    cfg=None,
    require_drive=True,
):
    cfg = copy.deepcopy(settings() if cfg is None else cfg)
    local = Path(local)
    if cfg["widths"] != [1024, 4096] or cfg["seed"] != 21:
        raise ValueError("Declared capacity probe changed")
    source = memory.FrozenStore(
        Path(source_output) / "checkpoints" / EXPECTED[:20], local / "read_source"
    )
    if source.run_id != EXPECTED or source.config["source_sha"] != EXPECTED_SOURCE:
        raise ValueError("Expected the completed 95e7186b193e GGF source")
    parent = memory.FrozenStore(
        Path(activation_output) / "checkpoints" / source.config["parent_run_id"][:20],
        local / "read_activations",
    )
    if parent.run_id != source.config["parent_run_id"]:
        raise ValueError("Wrong activation parent")
    for root in (Path(source_output).resolve(), Path(activation_output).resolve()):
        if root == Path(output).resolve() or root in Path(output).resolve().parents:
            raise ValueError("Output must be separate from each frozen source")
    base = copy.deepcopy(source.config["native_config"])
    audit = copy.deepcopy(source.config["config"])
    if (base["groups"], base["block_rank"], base["active_groups"], base["epochs"]) != (
        1024,
        3,
        16,
        40,
    ):
        raise ValueError("Source native recipe changed")
    if len(base["lrs"]) != 1:
        raise ValueError("Expected one historically selected reconstruction LR")
    provenance = json.loads((parent.results / "provenance.json").read_text())
    plan = json.loads((parent.results / "coverage_plan.json").read_text())
    if (
        identity({k: v for k, v in plan.items() if k != "fingerprint"})
        != plan["fingerprint"]
    ):
        raise ValueError("Coverage fingerprint differs")
    if (
        plan["splits"] != provenance["splits"]
        or parent.config["data_plan"] != plan["fingerprint"]
    ):
        raise ValueError("Activation plan identity differs")
    cohort = plan["coverage"]["category_indices"]
    if cohort != audit["category_indices"] or len(cohort) != 75:
        raise ValueError("Expected the fixed 75-category cohort")
    sets = [set(ids) for ids in provenance["splits"].values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Split overlap")
    norm = memory.required(parent, "normalization")
    if (
        np.shape(norm["mean"]) != (196, 768)
        or not np.isfinite(norm["mean"]).all()
        or not np.isfinite(norm["scale"])
        or norm["scale"] <= 0
    ):
        raise ValueError("Invalid frozen normalization")
    frozen = memory.required(source, "all_models_frozen")
    if not frozen["complete"]:
        raise ValueError("Source is incomplete")
    reference = next(
        r for r in frozen["models"] if r["seed"] == 21 and r["arm"] == "native"
    )
    state = memory.required(source, reference["key"])
    if (
        reference["field"] != "best"
        or training.weight_sha(state["best"]) != reference["weights_sha"]
    ):
        raise ValueError("Native reference weights changed")
    target = (2000 * 196 // base["batch"]) * base["epochs"]
    if (
        state["target"] != target
        or state["updates"] != target
        or state["seed"] != 21
        or state["lr"] != base["lrs"][0]
    ):
        raise ValueError("Native reference budget differs")
    del state
    policy = reuse_policy(source.config["numerical_versions"], versions)
    base.update(device=cfg["device"], methods=["vanilla"], seeds=[21])
    audit.update(
        device=cfg["device"],
        bootstrap=cfg["bootstrap"],
        fresh_test_seed=cfg["metric_seed"],
        audit_image_batch=1,
        preview_images=0,
    )
    store = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            native_config=base,
            audit_config=audit,
            source_sha=source_sha,
            numerical_versions=versions,
            parent_run_id=source.run_id,
            activation_run_id=parent.run_id,
            data_plan=plan["fingerprint"],
            reference_policy=policy,
        ),
        local / "rescue",
        require_drive,
    )
    packs = {}
    for split, count in (
        ("train", 2000),
        ("select", 400),
        ("mapping", 400),
        ("confirm", 400),
    ):
        store.event("verified_cache_loading", split=split)
        packs[split] = memory.normalized_pack(
            parent, split, provenance, norm, local, audit["extract_chunk"]
        )
        if len(packs[split]["ids"]) != count:
            raise ValueError("Unexpected split size: " + split)
    coverage = {
        split: coverage_check(packs[split], audit, provenance["categories"])
        for split in ("mapping", "confirm")
    }
    store.write("coverage_preflight.json", coverage)
    for split, rows in coverage.items():
        unsupported = [r for r in rows if not r["supported"]]
        if unsupported:
            raise ValueError(
                "Fixed-cohort evidence missing before training in "
                + split
                + ": "
                + str(unsupported)
            )
    training.measure = memory.streaming_measure
    ctx = dict(
        store=store,
        source=source,
        parent=parent,
        cfg=cfg,
        base=base,
        audit=audit,
        packs=packs,
        provenance=provenance,
        local=local,
        reference=reference,
        policy=policy,
        stamps=memory.source_snapshot([source, parent]),
    )
    store.write(
        "protocol.json",
        dict(
            literature="https://arxiv.org/html/2606.25234v1",
            paper_scope="Appendix C.1 tests G=4096 on DINOv3 ViT-B; our COCO schedule is held fixed, not a full paper replication",
            native_config=base,
            metric_config=audit,
            splits={k: v["ids"] for k, v in packs.items()},
            changes=["Dictionary group count only"],
            seed=21,
            reference_policy=policy,
            random_streams="Same seed and data/schedule; model-width-dependent initialization changes subsequent batch/noise draws",
            training="Unmodified Goodfire Vanilla reconstruction loss, no object labels or mask initialization",
            checkpoint="Minimum selection MSE at identical evaluation updates; no object-based model choice",
            readout="Mapping split selects detector and contrast groups separately; final readouts freeze before confirm inference",
            scope=cfg["evaluation_scope"],
            compute="Equal updates and input batches; wider model costs more FLOPs, memory and time",
            stopping="40 epochs; this tests a fixed budget, not guaranteed convergence or the optimal width",
        ),
    )
    return ctx


def model_config(ctx, groups):
    return dict(ctx["base"], groups=groups)


def load_weights(ctx, row):
    st = memory.required(
        ctx["source"] if row["origin"] == "saved" else ctx["store"], row["key"]
    )
    w = st[row["field"]]
    if training.weight_sha(w) != row["weights_sha"]:
        raise ValueError("Frozen weights changed")
    return w


def preflight(ctx):
    s = ctx["store"]
    out = s.load("preflight")
    if out is not None:
        return out
    cfg = ctx["base"]
    device = ctx["cfg"]["device"]
    x = ctx["packs"]["train"]["x"].reshape(-1, 768)
    rows = []
    for groups in (1024, 4096):
        if groups == 1024 and ctx["policy"]["reuse_1024"]:
            continue
        c = model_config(ctx, groups)
        c.update(methods=["vanilla"], seeds=[21], lrs=[cfg["lrs"][0]])
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        tt = time.monotonic()
        result = training.preflight(x, c, device)["measurements"][0]
        result.update(
            groups=groups,
            elapsed_seconds=time.monotonic() - tt,
            gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated())
            if device.startswith("cuda")
            else 0,
        )
        rows.append(result)
    if not all(
        np.isfinite(r["seconds_per_update"]) and r["seconds_per_update"] > 0
        for r in rows
    ):
        raise FloatingPointError("Invalid preflight timing")
    out = dict(
        numerical_ok=True,
        measurements=rows,
        rough_gradient_minutes=sum(r["gradient_minutes"] for r in rows),
        excludes="Cache loading, selection checks, mapping/confirmation, bootstrap and checkpoint IO; not a wall-time guarantee",
    )
    s.save("preflight", out)
    s.write("preflight.json", out)
    return out


def train(ctx, max_updates=None):
    s = ctx["store"]
    old = s.load("all_models_frozen")
    if old is not None:
        return old
    rows = []
    pack = ctx["packs"]
    device = ctx["cfg"]["device"]
    for groups in (1024, 4096):
        if groups == 1024 and ctx["policy"]["reuse_1024"]:
            item = copy.deepcopy(ctx["reference"])
            item.update(groups=groups, origin="saved")
            rows.append(item)
            s.event("saved_native_control_reused", groups=groups)
            continue
        key = f"vanilla_g{groups}_s21"
        c = model_config(ctx, groups)
        t = time.monotonic()
        state = training.fit(
            s,
            key,
            "vanilla",
            pack["train"]["x"].reshape(-1, 768),
            pack["select"]["x"].reshape(-1, 768),
            c,
            21,
            c["lrs"][0],
            device,
            max_updates,
        )
        if state["updates"] != state["target"]:
            return dict(
                complete=False,
                last_key=key,
                updates=state["updates"],
                target=state["target"],
            )
        rows.append(
            dict(
                groups=groups,
                seed=21,
                arm="native",
                key=key,
                field="best",
                origin="new",
                weights_sha=training.weight_sha(state["best"]),
                updates=state["updates"],
                best_update=state["best_update"],
                parameters=state["parameters"],
                gradient_seconds=state["seconds"],
                this_invocation_wall_seconds=time.monotonic() - t,
            )
        )
        del state
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    out = dict(complete=True, models=rows)
    s.save("all_models_frozen", out)
    s.write("all_models_frozen.json", out)
    memory.assert_immutable(ctx["stamps"])
    return out


@torch.no_grad()
def disk_audit(model, pack, cfg, device, folder, event=None):
    """Identical audit values to metrics.audit_arrays, stored in bounded local mmaps."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    n = len(pack["ids"])
    G = model.n_groups
    r = model.group_size
    objects = []
    weights = []
    by_image = [[] for _ in range(n)]
    skipped = dict(
        crowd_instances_already_excluded=True,
        too_little_foreground=0,
        too_little_background=0,
    )
    for i, mm in enumerate(pack["masks"]):
        known = mm["crowd"] <= cfg["negative_fraction"]
        for obj in mm["instances"]:
            fg = np.asarray(obj["fraction"], np.float64) * known
            bg = (
                mm["fraction"][:, obj["category"]].astype(float) / 255
                <= cfg["negative_fraction"]
            ) & known
            if fg.sum() < cfg["min_object_patch_mass"]:
                skipped["too_little_foreground"] += 1
                continue
            if bg.sum() < cfg["min_background_patches"]:
                skipped["too_little_background"] += 1
                continue
            by_image[i].append(len(objects))
            weights.append(fg / fg.sum() - bg / bg.sum())
            objects.append(
                dict(
                    image_id=int(pack["ids"][i]),
                    image_index=i,
                    annotation_id=obj["annotation_id"],
                    category=obj["category"],
                    foreground_mass=float(fg.sum()),
                    background_patches=int(bg.sum()),
                )
            )
    if not objects:
        raise ValueError("No eligible object contrasts")

    def mmap(name, shape, dtype):
        return np.lib.format.open_memmap(
            folder / (name + ".npy"), mode="w+", dtype=dtype, shape=shape
        )

    p = min(cfg["detection_patches"], 196)
    heat = mmap("heat", (n * p, G), np.float32)
    dz = mmap("dz", (len(objects), G, r), np.float32)
    dx = mmap("dx", (len(objects), model.d), np.float32)
    frac = []
    crowd = []
    model.eval()
    for i in range(n):
        xb = torch.as_tensor(np.array(pack["x"][i], copy=True), device=device)
        z = model.encode(xb).cpu().numpy().reshape(196, G, r)
        if not np.isfinite(z).all():
            raise FloatingPointError(
                "Nonfinite audit codes at image " + str(pack["ids"][i])
            )
        pick = np.random.default_rng(
            cfg["patch_sample_seed"] + int(pack["ids"][i])
        ).choice(196, p, replace=False)
        heat[i * p : (i + 1) * p] = np.linalg.norm(z, axis=-1)[pick]
        frac.append(pack["masks"][i]["fraction"][pick])
        crowd.append(pack["masks"][i]["crowd"][pick])
        for k in by_image[i]:
            dx[k] = weights[k] @ pack["x"][i]
            dz[k] = np.einsum("p,pgr->gr", weights[k], z)
        if event and (i + 1) % 40 == 0:
            event("audit_encoding", images=i + 1, total=n, groups=G)
    for a in (heat, dz, dx):
        a.flush()
    return dict(
        heat=heat,
        dz=dz,
        dx=dx,
        fraction=np.concatenate(frac),
        crowd=np.concatenate(crowd),
        image_index=np.repeat(np.arange(n), p),
        image_ids=pack["ids"],
        objects=objects,
        atoms=model.atoms().cpu().numpy().astype(np.float32),
        skipped=skipped,
    )


def calibrate(a, cfg, categories, chunk=64, event=None):
    """Exact one-group AP/capture selection; never materialize all decoded contributions."""
    y, v = metrics.binary_labels(a, cfg)
    G = a["atoms"].shape[0]
    C = len(categories)
    aps = np.full((G, C), np.nan)
    caps = np.full((G, C), np.nan)
    object_cats = np.array([o["category"] for o in a["objects"]])
    cohort = cfg["category_indices"]
    for lo in range(0, G, chunk):
        hi = min(lo + chunk, G)
        aps[lo:hi] = metrics.ap_matrix(a["heat"][:, lo:hi], y, v)
        D = a["atoms"][lo:hi].astype(np.float64)
        gram = np.einsum("grd,gsd->grs", D, D)
        for c in cohort:
            idx = np.flatnonzero(object_cats == c)
            dx = a["dx"][idx].astype(np.float64)
            z = np.asarray(a["dz"][idx, lo:hi], np.float64)
            den = float(np.square(dx).sum())
            if len(idx) and den > 1e-12:
                dot = np.einsum("md,grd->mgr", dx, D, optimize=True)
                gain = 2 * np.einsum("mgr,mgr->g", z, dot) - np.einsum(
                    "mgr,grs,mgs->g", z, gram, z
                )
                caps[lo:hi, c] = gain / den
        if event:
            event("readout_group_chunk", done=hi, total=G)
    mappings = []
    for c in cohort:
        p, n = metrics.image_counts(y[:, c], v[:, c], a["image_index"])
        idx = np.flatnonzero(object_cats == c)
        ni = len({a["objects"][i]["image_id"] for i in idx})
        if (
            min(p, n, ni) < cfg["min_category_images"]
            or not np.isfinite(aps[:, c]).any()
            or not np.isfinite(caps[:, c]).any()
        ):
            raise ValueError(
                "Insufficient mapping evidence for fixed category " + str(c)
            )
        dg = int(np.nanargmax(aps[:, c]))
        cg = int(np.nanargmax(caps[:, c]))
        mappings.append(
            dict(
                category_index=c,
                category=categories[c]["name"],
                category_id=categories[c]["id"],
                calibration_positive_images=p,
                calibration_negative_images=n,
                calibration_objects=len(idx),
                detector=dict(
                    group=dg,
                    threshold=metrics.f1_threshold(a["heat"][:, dg], y[:, c], v[:, c]),
                    calibration_ap=float(aps[dg, c]),
                ),
                contrast=dict(groups=[cg], calibration_capture=[float(caps[cg, c])]),
            )
        )
    return (
        dict(
            mappings=mappings,
            groups=G,
            rule="One-group readouts frozen using mapping data only",
        ),
        aps,
        caps,
    )


def score(a, mapping, cfg):
    compact = study.compact_audit(a, mapping, cfg)
    y, v = metrics.binary_labels(a, cfg)
    rows = []
    cats = np.array([o["category"] for o in a["objects"]])
    for j, item in enumerate(mapping["mappings"]):
        c = item["category_index"]
        p, n = metrics.image_counts(y[:, c], v[:, c], a["image_index"])
        idx = np.flatnonzero(cats == c)
        ni = len({a["objects"][i]["image_id"] for i in idx})
        den = float(compact["denominator"][:, j].sum())
        if min(p, n, ni) < cfg["min_category_images"] or den <= 1e-12:
            raise ValueError(
                "Insufficient confirmation evidence for fixed category " + str(c)
            )
        detection = metrics.detection(
            compact["detection_scores"][:, j],
            y[:, c],
            v[:, c],
            item["detector"]["threshold"],
        )
        capture = float(1 - compact["error"][:, j].sum() / den)
        if detection["ap"] is None or not np.isfinite([capture, detection["ap"]]).all():
            raise FloatingPointError("Invalid selected metric")
        rows.append(
            dict(
                category_index=c,
                category=item["category"],
                detector_group=item["detector"]["group"],
                contrast_group=item["contrast"]["groups"][0],
                single_group_contrast_capture=capture,
                detection=detection,
                objects=len(idx),
                object_images=ni,
                positive_images=p,
                negative_images=n,
            )
        )
    return dict(
        categories=rows,
        mean_single_group_contrast_capture=float(
            np.mean([r["single_group_contrast_capture"] for r in rows])
        ),
        mean_single_group_ap=float(np.mean([r["detection"]["ap"] for r in rows])),
    ), compact


def evaluate(ctx):
    s = ctx["store"]
    old = s.load("summary")
    if old is not None:
        outputs(ctx, old)
        return old
    catalog = memory.required(s, "all_models_frozen")
    reports = []
    for row in catalog["models"]:
        tag = f"g{row['groups']}"
        report = s.load("report_" + tag)
        if report is not None:
            reports.append(report)
            continue
        w = load_weights(ctx, row)
        m = training.make_model("vanilla", model_config(ctx, row["groups"]), 768).to(
            ctx["cfg"]["device"]
        )
        m.load_state_dict(w)
        del w
        mapping = s.load("mapping_" + tag)
        if mapping is None:
            s.event("mapping_start", groups=row["groups"])
            a = disk_audit(
                m,
                ctx["packs"]["mapping"],
                ctx["audit"],
                ctx["cfg"]["device"],
                ctx["local"] / tag / "mapping",
                s.event,
            )
            mapping, aps, caps = calibrate(
                a,
                ctx["audit"],
                ctx["provenance"]["categories"],
                ctx["cfg"]["group_chunk"],
                s.event,
            )
            np.savez_compressed(
                s.results / ("candidate_readouts_" + tag + ".npz"), ap=aps, capture=caps
            )
            # Candidate NaNs signify unsupported/out-of-cohort cells; never serialize them as JSON.
            s.save("mapping_" + tag, mapping)
            s.write("mappings/" + tag + ".json", mapping)
            del a, aps, caps
            gc.collect()
        s.event("confirmation_scoring_start", groups=row["groups"])
        a = disk_audit(
            m,
            ctx["packs"]["confirm"],
            ctx["audit"],
            ctx["cfg"]["device"],
            ctx["local"] / tag / "confirm",
            s.event,
        )
        report, compact = score(a, mapping, ctx["audit"])
        rec = memory.streaming_measure(
            m,
            ctx["packs"]["confirm"]["x"].reshape(-1, 768),
            ctx["cfg"]["device"],
            ctx["base"]["batch"],
        )
        report.update(model=row, reconstruction=rec, skipped_objects=a["skipped"])
        s.save("compact_" + tag, compact)
        s.save("report_" + tag, report)
        s.write("reports/" + tag + ".json", report)
        reports.append(report)
        del m, a, compact
        gc.collect()
        if ctx["cfg"]["device"].startswith("cuda"):
            torch.cuda.empty_cache()
    a, b = reports[1], reports[0]
    s.event("paired_bootstrap_start", replicates=ctx["cfg"]["bootstrap"])
    ci = study.paired(
        memory.required(s, "compact_g4096"),
        memory.required(s, "compact_g1024"),
        ctx["audit"],
    )
    summary = dict(
        models=reports,
        comparison=dict(
            capture_gain=a["mean_single_group_contrast_capture"]
            - b["mean_single_group_contrast_capture"],
            detection_ap_gain=a["mean_single_group_ap"] - b["mean_single_group_ap"],
            reconstruction_ve_gain=a["reconstruction"]["ve"]
            - b["reconstruction"]["ve"],
            paired_intervals=ci,
        ),
        scope=ctx["cfg"]["evaluation_scope"],
        seed=21,
        interpretation="Report each metric separately. Higher reconstruction alone does not establish better object grouping or an optimal width.",
    )
    memory.assert_immutable(ctx["stamps"])
    s.save("summary", summary)
    s.write("summary.json", summary)
    outputs(ctx, summary)
    s.event(
        "complete",
        **{k: v for k, v in summary["comparison"].items() if k != "paired_intervals"},
    )
    return summary


def outputs(ctx, summary):
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = ctx["store"]
    rows = []
    cats = []
    for report in summary["models"]:
        g = report["model"]["groups"]
        rec = report["reconstruction"]
        rows.append(
            dict(
                groups=g,
                seed=21,
                capture_percent=100 * report["mean_single_group_contrast_capture"],
                detection_ap_percent=100 * report["mean_single_group_ap"],
                reconstruction_ve_percent=100 * rec["ve"],
                reconstruction_mse=rec["mse"],
                active_groups=rec["active_groups"],
                dead_groups=rec["dead_groups"],
                parameters=report["model"].get("parameters", 2 * 768 * g * 3 + g * 3),
                selected_update=report["model"]["best_update"],
                updates=report["model"]["updates"],
                origin=report["model"]["origin"],
            )
        )
        for r in report["categories"]:
            cats.append(
                dict(
                    groups=g,
                    **{k: v for k, v in r.items() if k != "detection"},
                    **{"detection_" + k: v for k, v in r["detection"].items()},
                )
            )
    pd.DataFrame(rows).to_csv(s.results / "comparison.csv", index=False)
    pd.DataFrame(cats).to_csv(s.results / "per_category.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    for ax, col, title in zip(
        axes,
        ("capture_percent", "detection_ap_percent", "reconstruction_ve_percent"),
        ("Object contrast capture", "Patch detection AP", "Reconstruction VE"),
    ):
        vals = [r[col] for r in rows]
        ax.bar(["1,024", "4,096"], vals, color=["#607d8b", "#00897b"])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Percent")
        ax.set_ylim(min(0, min(vals) - 5), max(100, max(vals) + 5))
        for i, v in enumerate(vals):
            ax.text(i, v + 1, f"{v:.2f}", ha="center", fontsize=9)
    fig.savefig(s.results / "comparison.png", dpi=180)
    fig.savefig(s.results / "comparison.pdf")
    plt.close(fig)
    curves = []
    for row in memory.required(s, "all_models_frozen")["models"]:
        store = ctx["source"] if row["origin"] == "saved" else s
        st = memory.required(store, row["key"])
        for h in st["history"]:
            curves.append(
                dict(
                    groups=row["groups"],
                    update=h["update"],
                    selection_mse=h["selection"]["mse"],
                    selection_ve=h["selection"]["ve"],
                    seconds_per_update=h["seconds_per_update"],
                    training_loss=h["training_loss"],
                    lr=h["lr"],
                )
            )
    pd.DataFrame(curves).to_csv(s.results / "training_curves.csv", index=False)
    fig, ax = plt.subplots(figsize=(5, 3), layout="constrained")
    for g in (1024, 4096):
        rr = [r for r in curves if r["groups"] == g]
        ax.plot(
            [r["update"] for r in rr],
            [r["selection_mse"] for r in rr],
            label=f"{g:,} groups",
        )
    ax.set(xlabel="Training updates", ylabel="Selection reconstruction MSE")
    ax.legend()
    fig.savefig(s.results / "convergence.png", dpi=180)
    plt.close(fig)


def export(ctx):
    s = ctx["store"]
    s.check_mount()
    path = s.output / ("vanilla_bsf_width_" + s.run_id[:12] + "_results.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(s.results))
    return path
