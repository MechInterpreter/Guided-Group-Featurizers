"""Saved-start, fixed-recipe loss ablation. All parent artifacts are read-only."""

import copy
import gc
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

from coco_baselines import metrics, training
from coco_baselines.storage import identity
from coco_adaptation import core, inputs, run as adaptation
from coco_coordinated import run as memory
from coco_ggf import study

EXPECTED = "95e7186b193ed3744ced01585a0ff0f4a198c7318a53ab23cde20b4adb5082fd"
EXPECTED_SOURCE = "eb291dc32c7989cd3f64b5aa4cf46c8369ab6c7fd50b764f7f2f8a326371f7df"
ARMS = dict(contrast_only=(0.05, 0.0), detection_only=(0.0, 0.01), both=(0.05, 0.01))
LABELS = dict(
    native="Native Vanilla BSF start",
    continued_object="Continued BSF: object selection",
    contrast_only="GGF: contrast only",
    detection_only="GGF: detection only",
    both="GGF: both losses",
)


def settings():
    return dict(
        version=1,
        study="ggf_saved_start_loss_ablation",
        seeds=[21, 22, 23],
        variants=ARMS,
        allow_reference_replay=True,
        bootstrap=100,
        inherited_test="Already inspected 1500-image holdout; component diagnosis, not fresh confirmation",
    )


def reuse_policy(saved, current):
    differences = {
        k: dict(saved=v, current=current.get(k))
        for k, v in saved.items()
        if current.get(k) != v
    }
    return dict(
        reuse_references=not differences,
        version_differences=differences,
        decision="Reuse verified combined/control runs"
        if not differences
        else "Replay combined/control continuations from saved native starts; do not reinstall numeric packages",
    )


def fit_context(ctx, arm):
    c = dict(ctx)
    c["cfg"] = copy.deepcopy(ctx["cfg"])
    cap, det = ARMS.get(arm, (0.0, 0.0))
    c["cfg"].update(lambda_capture=cap, lambda_detection=det)
    return c


def prepare(
    source_output, output, local, source_sha, versions, cfg=None, require_drive=True
):
    local = Path(local)
    cfg = copy.deepcopy(settings() if cfg is None else cfg)
    source = memory.FrozenStore(
        Path(source_output) / "checkpoints" / EXPECTED[:20], local / "read_confirmation"
    )
    if source.run_id != EXPECTED or source.config["source_sha"] != EXPECTED_SOURCE:
        raise ValueError("Wrong completed GGF source; no retraining fallback")
    root = Path(source_output).resolve()
    dest = Path(output).resolve()
    if dest == root or root in dest.parents:
        raise ValueError("Output must be separate from frozen source")
    original = memory.FrozenStore(
        Path(source_output).parent
        / "AGF_COCO_stratified_confirmation_v1"
        / "checkpoints"
        / source.config["parent_run_id"][:20],
        local / "read_activations",
    )
    if original.run_id != source.config["parent_run_id"]:
        raise ValueError("Wrong activation parent")
    frozen = memory.required(source, "all_models_frozen")
    if (
        not frozen["complete"]
        or sorted(set(r["seed"] for r in frozen["models"])) != cfg["seeds"]
    ):
        raise ValueError("Expected all three saved starts")
    scfg = copy.deepcopy(source.config["config"])
    base = copy.deepcopy(source.config["native_config"])
    if (scfg["groups"], scfg["block_rank"], scfg["active_groups"], scfg["updates"]) != (
        1024,
        3,
        16,
        2500,
    ):
        raise ValueError("Successful source recipe differs")
    if cfg["variants"] != ARMS:
        raise ValueError(
            "Freeze the declared ablation weights, not a hyperparameter search"
        )
    policy = reuse_policy(source.config["numerical_versions"], versions)
    if not policy["reuse_references"] and not cfg["allow_reference_replay"]:
        raise ValueError(
            "Reference replay required for changed versions; enable it rather than mix environments"
        )
    store = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=source_sha,
            numerical_versions=versions,
            parent_run_id=source.run_id,
            source_config=source.config,
            reference_policy=policy,
        ),
        local / "rescue",
        require_drive,
    )
    memory.memory_safe_metrics()
    provenance = json.loads((original.results / "provenance.json").read_text())
    norm = memory.required(original, "normalization")
    plan = memory.required(source, "fresh_plan")
    sets = [set(ids) for ids in provenance["splits"].values()]
    if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
        raise ValueError("Historical split overlap")
    if set(plan["splits"]["test"]) & set().union(*sets):
        raise ValueError("Inherited test overlaps a historical split")
    if len(plan["splits"]["test"]) != 1500:
        raise ValueError("Wrong inherited holdout")
    packs = {}
    for split in ("train", "select", "mapping"):
        store.event("loading_verified_activation_cache", split=split)
        packs[split] = memory.normalized_pack(
            original, split, provenance, norm, local, scfg["extract_chunk"]
        )
    targets = {}
    checks = []
    for seed in cfg["seeds"]:
        t = memory.required(source, f"targets_s{seed}")
        if sorted(r["category_index"] for r in t) != scfg["category_indices"]:
            raise ValueError("Saved target cohort changed")
        targets[seed] = t
        native = next(
            r for r in frozen["models"] if r["seed"] == seed and r["arm"] == "native"
        )
        st = memory.required(source, native["key"])
        if training.weight_sha(st[native["field"]]) != native["weights_sha"]:
            raise ValueError("Native checkpoint hash differs")
        del st
        pair = json.loads(
            (source.results / f"paired_training_s{seed}.json").read_text()
        )
        if (
            pair["common_start_sha"] != native["weights_sha"]
            or pair["matched_gradient_updates"] != 2500
        ):
            raise ValueError("Historical paired protocol mismatch")
        checks.append(pair)
    ctx = dict(
        store=store,
        source=source,
        original=original,
        cfg=scfg,
        base=base,
        packs=packs,
        provenance=provenance,
        norm=norm,
        plan=plan,
        targets=targets,
        local=local,
        ablation=cfg,
        policy=policy,
        source_frozen=frozen,
        stamps=memory.source_snapshot([source, original]),
    )
    store.write("reference_policy.json", policy)
    store.write(
        "protocol_settings.json",
        dict(
            ablation=cfg,
            source_recipe=scfg,
            source_training=checks,
            fixed_assignments="Reuse source training targets; no remapping before learning",
            checkpoint_rule="Every arm uses capture + 0.2 AP under the original 2% relative MSE guard",
            no_new_data="No DINO extraction, no native pretraining, no new test sampling",
            evaluation_scope="Known test-set diagnostic; later independent confirmation remains necessary",
        ),
    )
    return ctx


def descriptor(seed, arm, state, key, origin="new", source_tag=None):
    result = dict(
        seed=seed,
        arm=arm,
        key=key,
        field="best",
        updates=state["updates"],
        best_update=state["best_update"],
        weights_sha=training.weight_sha(state["best"]),
        parameters=sum(v.numel() for v in state["best"].values()),
        auxiliary_training_head_parameters=state["head"].numel()
        if arm in ("detection_only", "both")
        else 0,
        origin=origin,
        source_tag=source_tag,
    )
    return result


def preflight(ctx):
    s = ctx["store"]
    old = s.load("preflight")
    if old is not None:
        return old
    cfg = ctx["cfg"]
    device = cfg["device"]
    seed = ctx["ablation"]["seeds"][0]
    st = memory.required(ctx["source"], f"native_s{seed}")
    initial = st["best"]
    del st
    times = {}
    last = None
    arms = ["contrast_only", "detection_only"] + (
        [] if ctx["policy"]["reuse_references"] else ["both", "continued_object"]
    )
    for arm in arms:
        m = core.new_model("vanilla", ctx["base"], initial, device)
        head = torch.nn.Parameter(
            torch.zeros(len(ctx["targets"][seed]), 2, device=device)
        )
        opt = torch.optim.Adam(
            list(m.parameters()) + ([head] if arm != "continued_object" else []),
            lr=cfg["lr"],
        )
        ac = fit_context(ctx, arm)["cfg"]
        values = []
        for i in range(6):
            ix = np.arange(cfg["images_per_update"])
            x = torch.as_tensor(
                np.array(ctx["packs"]["train"]["x"][ix], copy=True), device=device
            )
            study.sync(device)
            tt = time.monotonic()
            xx = x.reshape(-1, m.d)
            noisy = (
                xx + cfg["snr"] * xx.std() * torch.randn_like(xx) if cfg["snr"] else xx
            )
            native, _ = m.loss(noisy, target=xx)
            loss = native
            if arm != "continued_object":
                cap, det, _ = core.supervised_loss(
                    m,
                    x,
                    [ctx["packs"]["train"]["masks"][j] for j in ix],
                    ctx["targets"][seed],
                    head,
                    ac,
                )
                loss = loss + ac["lambda_capture"] * cap + ac["lambda_detection"] * det
            if not torch.isfinite(loss):
                raise FloatingPointError("Invalid preflight loss")
            opt.zero_grad()
            loss.backward()
            opt.step()
            m.normalize_decoder()
            study.sync(device)
            if i >= 2:
                values.append(time.monotonic() - tt)
        times[arm] = float(np.median(values))
        last = m
        del head, opt
        if arm != arms[-1]:
            del m
            gc.collect()
    tt = time.monotonic()
    validation = core.validation(
        last, ctx["packs"]["select"], ctx["targets"][seed], cfg, device
    )
    validation_seconds = time.monotonic() - tt
    tt = time.monotonic()
    s.save("timing_checkpoint", dict(weights=training.weights(last)))
    save_seconds = time.monotonic() - tt
    n = len(ctx["ablation"]["seeds"])
    calls = int(np.ceil(cfg["updates"] / cfg["evaluate_every"])) + 1
    seconds = n * (
        sum(times.values()) * cfg["updates"]
        + len(arms) * calls * (validation_seconds + save_seconds)
        + len(arms) * validation_seconds * 5
    )
    out = dict(
        new_training_fits=n * len(arms),
        saved_native_starts=3,
        reused_test_images=1500,
        seconds_per_update=times,
        full_selection_seconds=validation_seconds,
        drive_save_seconds=save_seconds,
        rough_total_hours=[seconds / 3600, 2 * seconds / 3600],
        reference_reuse=ctx["policy"]["reuse_references"],
        numerical_ok=all(
            np.isfinite(validation["reconstruction"][k]) for k in ("mse", "ve")
        ),
        excludes="Drive stalls, cache loading, annotation audits and bootstrap variability; scenarios, not guaranteed ETA",
    )
    s.save("preflight", out)
    s.write("preflight.json", out)
    del last, m, initial
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def train(ctx, max_updates=None):
    s = ctx["store"]
    old = s.load("all_models_frozen")
    if old is not None:
        return old
    rows = []
    for seed in ctx["ablation"]["seeds"]:
        parent = ctx["source"]
        st = memory.required(parent, f"native_s{seed}")
        initial = st["best"]
        del st
        native = copy.deepcopy(
            next(
                r
                for r in ctx["source_frozen"]["models"]
                if r["seed"] == seed and r["arm"] == "native"
            )
        )
        native.update(origin="saved", source_tag=f"s{seed}_native")
        rows.append(native)
        expected_batch = json.loads(
            (parent.results / f"paired_training_s{seed}.json").read_text()
        )["minibatches_sha"]
        for arm in ("continued_object", "contrast_only", "detection_only", "both"):
            if ctx["policy"]["reuse_references"] and arm in (
                "continued_object",
                "both",
            ):
                srcarm = "ggf" if arm == "both" else arm
                item = copy.deepcopy(
                    next(
                        r
                        for r in ctx["source_frozen"]["models"]
                        if r["seed"] == seed and r["arm"] == srcarm
                    )
                )
                oldstate = memory.required(parent, item["key"])
                if (
                    oldstate["initial_sha"] != native["weights_sha"]
                    or oldstate["sampled_batches_sha"] != expected_batch
                ):
                    raise ValueError("Reference start/minibatches changed")
                if training.weight_sha(oldstate[item["field"]]) != item["weights_sha"]:
                    raise ValueError("Reference weights changed")
                del oldstate
                item.update(arm=arm, origin="saved", source_tag=f"s{seed}_{srcarm}")
                rows.append(item)
                s.event(
                    "reused_completed_reference", seed=seed, arm=arm, key=item["key"]
                )
                continue
            key = f"s{seed}_{arm}"
            s.event("ablation_fit_start", seed=seed, arm=arm, key=key)
            state = study.fit(
                fit_context(ctx, arm),
                key,
                initial,
                ctx["targets"][seed],
                arm != "continued_object",
                71000 + seed,
                max_updates=max_updates,
            )
            if max_updates is not None and state["updates"] < ctx["cfg"]["updates"]:
                return dict(complete=False)
            if state["sampled_batches_sha"] != expected_batch:
                raise ValueError("Ablation minibatches differ from paired source")
            item = descriptor(seed, arm, state, key)
            item["parameters"] = native["parameters"]
            rows.append(item)
            s.write(
                f"paired_training_s{seed}_{arm}.json",
                dict(
                    common_start_sha=native["weights_sha"],
                    minibatches_sha=expected_batch,
                    updates=state["updates"],
                    loss_weights=ARMS.get(arm, (0.0, 0.0)),
                    best_update=state["best_update"],
                    target_sha=state["target_sha"],
                    checkpoint_guard=ctx["cfg"]["fit_reconstruction_tolerance"],
                ),
            )
            del state
            gc.collect()
        del initial
        gc.collect()
    out = dict(
        complete=True,
        models=rows,
        test_used=False,
        structural_edits=0,
        dimensions_changed=False,
    )
    s.save("all_models_frozen", out)
    s.write("all_models_frozen.json", out)
    memory.assert_immutable(ctx["stamps"])
    return out


def load_test(ctx):
    frozen = memory.required(ctx["store"], "all_models_frozen")
    if not frozen["complete"]:
        raise ValueError("Freeze every ablation before opening test")
    p = copy.deepcopy(ctx["provenance"])
    p["splits"]["test"] = ctx["plan"]["splits"]["test"]
    ctx["store"].event("loading_saved_test_cache", images=len(p["splits"]["test"]))
    return memory.normalized_pack(
        ctx["source"], "test", p, ctx["norm"], ctx["local"], ctx["cfg"]["extract_chunk"]
    )


def compact(ctx, item):
    source = ctx["source"] if item["origin"] == "saved" else ctx["store"]
    tag = (
        item["source_tag"]
        if item["origin"] == "saved"
        else f"s{item['seed']}_{item['arm']}"
    )
    return memory.required(source, "compact_" + tag)


def evaluate(ctx, test):
    s, cfg = ctx["store"], ctx["cfg"]
    frozen = memory.required(s, "all_models_frozen")
    reports = []
    for item in frozen["models"]:
        tag = f"s{item['seed']}_{item['arm']}"
        cached = s.load("report_" + tag)
        if cached is not None:
            reports.append(cached)
            print("Reused audit", tag, flush=True)
            continue
        if item["origin"] == "saved":
            r = copy.deepcopy(
                memory.required(ctx["source"], "report_" + item["source_tag"])
            )
            if r["summary"]["weights_sha"] != item["weights_sha"]:
                raise ValueError("Imported report weight hash differs")
            r["summary"].update(**item, label=LABELS[item["arm"]])
            r["readout_origin"] = "Verified saved confirmation report"
            s.save("report_" + tag, r)
            s.write("reports/" + tag + ".json", r)
            reports.append(r)
            continue
        st = memory.required(s, item["key"])
        w = st[item["field"]]
        del st
        if training.weight_sha(w) != item["weights_sha"]:
            raise ValueError("Frozen ablation changed")
        m = core.new_model("vanilla", ctx["base"], w, cfg["device"])
        del w
        mapping = s.load("mapping_" + tag)
        if mapping is None:
            s.event("calibration_start", model=tag)
            p = ctx["packs"]["mapping"]
            a = metrics.audit_arrays(
                m, p["x"], p["masks"], p["ids"], cfg, cfg["device"]
            )
            mapping = metrics.calibrate(a, cfg, ctx["provenance"]["categories"])
            del a
            gc.collect()
            s.save("mapping_" + tag, mapping)
            s.write(
                "mappings/" + tag + ".json",
                dict(mappings=mapping["mappings"], weights_sha=item["weights_sha"]),
            )
        s.event("test_scoring_start", model=tag)
        audit = metrics.audit_arrays(
            m, test["x"], test["masks"], test["ids"], cfg, cfg["device"]
        )
        categories = metrics.evaluate(
            audit, mapping, dict(cfg, bootstrap=0), cfg["fresh_test_seed"]
        )
        indexed = {r["category_index"]: r for r in categories}
        selected = [indexed[c] for c in cfg["category_indices"]]
        if any(r["contrast"] is None or r["detection"] is None for r in selected):
            raise ValueError("Missing fixed category evidence")
        reconstruction = training.measure(
            m, test["x"].reshape(-1, m.d), cfg["device"], cfg["batch"]
        )
        ca = study.compact_audit(audit, mapping, cfg)
        s.save("compact_" + tag, ca)
        summary = dict(
            **item,
            method="vanilla",
            label=LABELS[item["arm"]],
            groups=cfg["groups"],
            block_rank=cfg["block_rank"],
            reconstruction=reconstruction,
            categories=len(selected),
            images=len(test["ids"]),
            structural_edits=0,
            dimensions_changed=False,
        )
        for field, path in [
            (
                "mean_single_group_contrast_capture",
                ("contrast", "single_group_capture"),
            ),
            ("mean_single_group_ap", ("detection", "ap")),
            ("mean_single_group_iou", ("detection", "iou")),
            ("mean_multi_group_contrast_capture", ("contrast", "multi_group_capture")),
            ("mean_all_group_contrast_capture", ("contrast", "all_group_capture")),
        ]:
            summary[field] = float(np.mean([r[path[0]][path[1]] for r in selected]))
        report = dict(
            summary=summary, categories=categories, skipped_objects=audit["skipped"]
        )
        s.save("report_" + tag, report)
        s.write("reports/" + tag + ".json", report)
        reports.append(report)
        s.event("model_audit_complete", **summary)
        del m, audit, ca
        gc.collect()
        if cfg["device"].startswith("cuda"):
            torch.cuda.empty_cache()
    comparisons = []
    pairs = [
        ("contrast_only", "continued_object"),
        ("detection_only", "continued_object"),
        ("both", "continued_object"),
        ("both", "contrast_only"),
        ("both", "detection_only"),
        ("contrast_only", "detection_only"),
    ]
    for seed in ctx["ablation"]["seeds"]:
        r = {
            p["summary"]["arm"]: p["summary"]
            for p in reports
            if p["summary"]["seed"] == seed
        }
        items = {v["arm"]: v for v in frozen["models"] if v["seed"] == seed}
        for a, b in pairs:
            key = f"paired_s{seed}_{a}_vs_{b}"
            ci = s.load(key)
            if ci is None:
                s.event("paired_bootstrap_start", seed=seed, arm=a, control=b)
                aa = compact(ctx, items[a])
                bb = compact(ctx, items[b])
                ci = study.paired(
                    aa, bb, dict(cfg, bootstrap=ctx["ablation"]["bootstrap"])
                )
                s.save(key, ci)
                del aa, bb
                gc.collect()
            gain = (
                r[a]["mean_single_group_contrast_capture"]
                - r[b]["mean_single_group_contrast_capture"]
            )
            apgain = r[a]["mean_single_group_ap"] - r[b]["mean_single_group_ap"]
            vedrop = r[b]["reconstruction"]["ve"] - r[a]["reconstruction"]["ve"]
            comparisons.append(
                dict(
                    seed=seed,
                    arm=a,
                    control=b,
                    capture_gain=gain,
                    ap_gain=apgain,
                    reconstruction_ve_drop=vedrop,
                    positive_within_predeclared_cost=gain > 0
                    and apgain >= 0
                    and vedrop <= cfg["max_reconstruction_ve_drop"],
                    paired_intervals=ci,
                )
            )
    summary = dict(
        models=[r["summary"] for r in reports],
        comparisons=comparisons,
        reference_policy=ctx["policy"],
        scope="Fixed-recipe diagnostic using saved native starts and a previously inspected holdout. All arms use the same checkpoint rule. Not fresh independent confirmation or proof of semantic purity.",
        interpretation="Both vs contrast-only isolates adding detection; both vs detection-only isolates adding contrast. Compare both capture and AP and the reconstruction cost; do not select solely on one favorable number.",
        no_claim="No structural adaptation, exact splitting rate, downstream task evaluation or supervised competitor reproduction in this notebook.",
    )
    s.write("summary.json", summary)
    adaptation.export_categories(s, reports)
    plots(ctx, reports)
    diagnostics(ctx, frozen, reports)
    memory.assert_immutable(ctx["stamps"])
    return reports, summary


def diagnostics(ctx, frozen, reports):
    s = ctx["store"]
    out = []
    for item in frozen["models"]:
        if item["arm"] == "native":
            continue
        src = ctx["source"] if item["origin"] == "saved" else s
        st = memory.required(src, item["key"])
        history = st["history"]
        last = history[-1]["validation"]
        picked = next(
            r["validation"] for r in history if r["update"] == st["best_update"]
        )
        grads = [
            r["component_gradients"] for r in history if r.get("component_gradients")
        ]
        out.append(
            dict(
                seed=item["seed"],
                arm=item["arm"],
                origin=item["origin"],
                selected_update=st["best_update"],
                total_updates=st["updates"],
                reconstruction_guard_rejections=sum(
                    r.get("object_checkpoint_eligible") is False for r in history
                ),
                selected_validation={
                    k: v for k, v in picked.items() if k != "categories"
                },
                last_validation={k: v for k, v in last.items() if k != "categories"},
                selected_categories=picked["categories"],
                last_categories=last["categories"],
                gradient_diagnostics=grads,
                inactive_loss="Detection has zero weight; its allocated helper head stays unused"
                if item["arm"] == "contrast_only"
                else None,
            )
        )
        s.write("histories/" + item["key"] + ".json", history)
        del st
    for seed in ctx["ablation"]["seeds"]:
        groupings = {}
        for field in ("contrast_group", "detector_group"):
            g = {}
            for t in ctx["targets"][seed]:
                g.setdefault(t[field], []).append(t["category"])
            groupings[field] = [
                dict(group=int(k), categories=v) for k, v in g.items() if len(v) > 1
            ]
        s.write(f"target_collisions_s{seed}.json", groupings)
    s.write(
        "component_diagnostics.json",
        dict(
            arms=out,
            note="Logs expose measured loss/gradient conflicts, guard rejection, late deterioration and mapping collisions; they cannot identify every causal failure automatically.",
        ),
    )


def plots(ctx, reports):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    folder = ctx["store"].results / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    arms = list(LABELS)
    for ax, field, title in zip(
        axes,
        ["mean_single_group_contrast_capture", "mean_single_group_ap", "ve"],
        ["Object contrast capture", "Detection AP", "Reconstruction VE"],
    ):
        for j, arm in enumerate(arms):
            vals = [
                100
                * (
                    r["summary"]["reconstruction"]["ve"]
                    if field == "ve"
                    else r["summary"][field]
                )
                for r in reports
                if r["summary"]["arm"] == arm
            ]
            ax.bar(
                j,
                np.mean(vals),
                color=["#bbc8d0", "#9eb4c4", "#ab8bc2", "#72b9a2", "#008d83"][j],
            )
            ax.scatter(
                np.linspace(j - 0.1, j + 0.1, len(vals)),
                vals,
                color="black",
                s=25,
                zorder=3,
            )
        ax.set_xticks(
            range(len(arms)),
            [LABELS[k] for k in arms],
            rotation=28,
            ha="right",
            fontsize=8,
        )
        ax.set(title=title, ylabel="%")
        ax.grid(axis="y", alpha=0.15)
        ax.axhline(0, color="#444", lw=0.5)
    fig.suptitle("GGF loss ablation: all three starts, matched continued BSF")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"loss_ablation.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    frozen = memory.required(ctx["store"], "all_models_frozen")["models"]
    for i, seed in enumerate(ctx["ablation"]["seeds"]):
        for arm in ("continued_object", "contrast_only", "detection_only", "both"):
            item = next(r for r in frozen if r["seed"] == seed and r["arm"] == arm)
            src = ctx["source"] if item["origin"] == "saved" else ctx["store"]
            state = memory.required(src, item["key"])
            h = state["history"]
            for ax, field in zip(axes[i], ("macro_capture", "macro_ap", "ve")):
                y = [
                    100
                    * (
                        v["validation"]["reconstruction"]["ve"]
                        if field == "ve"
                        else v["validation"][field]
                    )
                    for v in h
                ]
                ax.plot([v["update"] for v in h], y, label=LABELS[arm])
                ax.set(title=f"Seed {seed}: {field}", xlabel="Updates", ylabel="%")
                ax.grid(alpha=0.15)
            del state
        axes[i, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(folder / "learning_curves.png", dpi=160)
    plt.close(fig)


def export(ctx, source):
    s = ctx["store"]
    path = s.rescue / f"ggf_loss_ablation_{s.run_id[:12]}_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in s.results.rglob("*"):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(s.results).as_posix())
        for p in Path(source).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, "source/" + p.relative_to(source).as_posix())
    return path
