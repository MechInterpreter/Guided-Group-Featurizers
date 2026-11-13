"""Frozen dependency-guided AGF across saved starts and a new COCO holdout.

No native baseline retraining. Two fixed-horizon GGF continuations, then the
unchanged mature-regroup recipe. Every candidate freezes before fresh scoring.
"""

import copy
import csv
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from coco_adaptation import inputs, coverage
from coco_baselines import data, metrics, training
from coco_baselines.storage import identity, sha_file
from coco_coordinated import run as memory
from coco_stratified import dataset
from . import mature_regroup as mature, accumulating_regroup as accum
from . import direction_regroup as prior, bank_experiment as bank
from . import owner_routing as old, study

MATURE_RUN = "b4ce459915c5827a53ad6d49024f99ba25bcc1008aa47c957024a53052c6a2d5"
MATURE_SHA = "9ccaa3892ea7f1df6e469e1cc641f8c35e3c5105569b87d62086870231497cee"


def settings():
    return dict(
        version=1,
        study="frozen_agf_independent_start_confirmation",
        seeds=[21, 22, 23],
        primary_seeds=[22, 23],
        reference_seed=21,
        common_continuation_updates=600,
        fresh_images=1500,
        fresh_min_images=20,
        fresh_seed=106051,
        bootstrap=200,
        extract_chunk=20,
        extract_batch=8,
        fresh_coverage_extension_limit=500,
        recipe=mature.settings(),
        scope="Saved independent dictionary starts on the same training data; seed 21 is a previously inspected reference. Starts 22/23 are the primary adaptation replications. One newly allocated, coverage-enriched test set; no candidate retries after test.",
    )


def final_start(ctx, seed):
    """Pair final dictionary with its actual final head, never an earlier best."""
    row = next(
        r for r in ctx["frozen"]["models"] if r["seed"] == seed and r["arm"] == "ggf"
    )
    st = memory.required(ctx["source"], row["key"])
    if st["updates"] != row["updates"] or st["updates"] != 2500:
        raise ValueError("Incomplete saved GGF start " + str(seed))
    targets = memory.required(ctx["source"], f"targets_s{seed}")
    if (
        sorted(t["category_index"] for t in targets)
        != ctx["eval_cfg"]["category_indices"]
    ):
        raise ValueError("Start changed the frozen category denominator")
    # Pair the stored final head with the final dictionary weights.
    w = st["weights"]
    h = st["head"]
    snap = bank.native_snapshot(ctx, w, targets, h, 1024, 3, 16)
    provenance = dict(
        seed=seed,
        key=row["key"],
        field="weights",
        updates=st["updates"],
        historical_best_update=st["best_update"],
        historical_best_sha=row["weights_sha"],
        final_dictionary_sha=training.weight_sha(w),
        final_snapshot_sha=accum.weight_sha(snap),
        head_policy="Saved final weights and saved final head; no best/final mismatch",
    )
    return snap, provenance


def prepare(
    parent, mature_parent, output, local, sha, versions, device, require_drive=True
):
    output = Path(output)
    local = Path(local)
    cfg = settings()
    for source in (Path(parent), Path(mature_parent)):
        a = output.resolve()
        b = source.resolve()
        if a == b or a in b.parents or b in a.parents:
            raise ValueError("Output overlaps a read-only parent")
    ctx = old.prepare(
        parent,
        output / "inputs",
        local,
        sha,
        versions,
        device=device,
        require_drive=require_drive,
    )
    src = memory.FrozenStore(
        Path(mature_parent) / "checkpoints" / MATURE_RUN[:20], local / "read_mature"
    )
    if src.run_id != MATURE_RUN or src.config["source_sha"] != MATURE_SHA:
        raise ValueError("Wrong completed mature-regroup parent")
    summary = json.loads((src.results / "summary.json").read_text())
    if not summary["complete"]:
        raise ValueError("Mature diagnostic is incomplete")
    snap = memory.required(src, "verified_start")
    if accum.weight_sha(snap) != src.config["start_weights_sha"]:
        raise ValueError("Mature root hash changed")
    roles = json.loads((src.results / "split_roles.json").read_text())
    for split in ("train", "select"):
        if roles["ids"][split] != ctx["packs"][split]["ids"]:
            raise ValueError("Mature parent data changed")
    ctx.update(
        study_cfg=cfg,
        mature_source=src,
        mature_start=snap,
        output=output,
        source_sha=sha,
        versions=versions,
        require_drive=require_drive,
        arms={},
    )
    ctx["stamps"] += memory.source_snapshot([src])
    ctx["store"] = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=sha,
            versions=versions,
            device=device,
            parent_run=ctx["source"].run_id,
            mature_run=MATURE_RUN,
            activation_run=ctx["original"].run_id,
        ),
        local / "replication_rescue",
        require_drive,
    )
    ctx["norm"] = memory.required(ctx["original"], "normalization")
    starts = {}
    for seed in cfg["seeds"]:
        initial, provenance = final_start(ctx, seed)
        rule = dict(cfg["recipe"], seed=seed, scope=cfg["scope"])
        store = inputs.JournalStore(
            output / f"seed_{seed}",
            dict(
                config=rule,
                source_sha=sha,
                versions=versions,
                device=device,
                source=provenance,
                mature_run=MATURE_RUN if seed == 21 else None,
            ),
            local / f"seed_{seed}_rescue",
            require_drive,
        )
        arm = dict(ctx, store=store, initial=initial, protocol=rule, prior=None)
        arm["cfg"] = dict(
            ctx["eval_cfg"],
            **{
                k: rule[k]
                for k in (
                    "lr",
                    "snr",
                    "lambda_capture",
                    "lambda_detection",
                    "grad_clip",
                    "images_per_update",
                )
            },
        )
        arm["source_weights_sha"] = accum.weight_sha(initial)
        ctx["arms"][seed] = arm
        starts[seed] = provenance
    if identity(snap["spec"]) != identity(ctx["arms"][21]["initial"]["spec"]):
        raise ValueError(
            "Mature seed-21 root changed the saved capacity or category readouts"
        )
    ctx["store"].write(
        "protocol_settings.json",
        dict(
            settings=cfg,
            starting_models=starts,
            gradients="Only seeds 22/23 receive 600 fixed-horizon GGF updates. AGF candidates receive zero gradient updates.",
            head_policy="Each continuation starts with matching final weights/head. The endpoint is u600, not the best development checkpoint.",
            inference="Native signed encoder, Top-16 group gate and decoder; no masks needed at inference.",
            budget="Shared maturity cost is recorded separately from AGF profiling, graph, scoring and extraction cost. Zero edit gradients is not zero compute.",
            frozen_recipe="Same pool 64, shortlist 8, max five exchanges and original preservation tolerances; no threshold tuning on test.",
            test_rule="Score every trained batch, even one failing development, alongside its exact root. Deployed policy separately falls back to the root after development rejection.",
        ),
    )
    return ctx


def fresh_plan(ctx, coco):
    cfg = ctx["study_cfg"]
    s = ctx["store"]
    saved = s.load("fresh_plan")
    if saved is not None:
        return saved
    previous = memory.required(ctx["source"], "fresh_plan")
    provenance = copy.deepcopy(ctx["provenance"])
    provenance["splits"]["prior_ggf_fresh_test"] = previous["splits"]["test"]
    planning = dict(
        ctx,
        store=s,
        parent=ctx["original"],
        parent_output=ctx["original"].output,
        provenance=provenance,
        cfg=dict(
            ctx["eval_cfg"],
            fresh_test_images=cfg["fresh_images"],
            fresh_test_seed=cfg["fresh_seed"],
            fresh_coverage_extension_limit=cfg["fresh_coverage_extension_limit"],
            gallery_seed=106052,
            gallery_categories=0,
        ),
    )
    plan = study.fresh_plan(planning, coco)
    used = set(i for ids in provenance["splits"].values() for i in ids)
    check_disjoint(plan["splits"]["test"], used)
    if plan["coverage"]["minimums"]["test"] != cfg["fresh_min_images"]:
        raise ValueError("Fresh coverage quota differs")
    plan["excluded_image_ids_sha"] = identity(sorted(used))
    plan["rule"] = (
        "New 1500-image mask-only plan; excludes all original splits and the previously scored GGF 1500-image test. Same 75-category cohort, minimum 20 usable images/category."
    )
    s.save("fresh_plan", plan)
    s.write("fresh_test_plan.json", plan)
    return plan


def check_disjoint(ids, used):
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate fresh image IDs")
    if set(ids) & set(used):
        raise ValueError("Fresh images overlap a historical split")


def freeze_all(ctx):
    """Fixed endpoint maturity; one bounded training-chosen batch per start."""
    master = ctx["store"]
    saved = master.load("all_models_frozen")
    if saved is not None:
        return saved
    rows = []
    for seed, arm in ctx["arms"].items():
        s = arm["store"]
        done = s.load("seed_frozen")
        if done is not None:
            rows.append(done)
            continue
        master.event(
            "seed_started",
            seed=seed,
            seed_progress=str(s.results / "progress.json"),
            role="reference" if seed == 21 else "primary independent start",
        )
        start = time.monotonic()
        s.event("seed_start", seed=seed)
        s.save("initial_preflight", prior.preflight(arm))
        if seed == 21:
            strong = ctx["mature_start"]
            maturity_updates = 0
        else:
            state = prior.fit(arm, "common_maturity", arm["initial"], updates=600)
            if state["updates"] != 600:
                raise ValueError("Continuation endpoint incomplete")
            strong = memory.required(s, "common_maturity_u600")
            maturity_updates = 600
            if accum.weight_sha(strong) != accum.weight_sha(state["current"]):
                raise ValueError("Endpoint snapshot mismatch")
        arm["initial"] = strong
        arm["source_weights_sha"] = accum.weight_sha(strong)
        s.save("verified_start", strong)
        s.save("accumulation_root", strong)
        prior.preflight(arm)
        mature.screen_and_graph(arm)
        batch = mature.accumulate(arm)
        edit = memory.required(s, "accumulated_snapshot")
        # Development makes one deploy/fallback decision, never chooses a prefix.
        control = accum.measured(arm, "select_original", strong, "select")
        candidate = accum.measured(arm, "select_accumulated", edit, "select")
        gate = accum.positive_gate(candidate, control, control, arm["protocol"])
        eligible = bool(batch["accepted_proposals"]) and gate["passed"]
        nomination = dict(
            seed=seed,
            root="verified_start",
            candidate="accumulated_snapshot",
            root_sha=accum.weight_sha(strong),
            candidate_sha=accum.weight_sha(edit),
            deploy_candidate=eligible,
            development_gate=gate,
            training_accepted_exchanges=len(batch["accepted_proposals"]),
            maturity_new_updates=maturity_updates,
            edit_gradient_updates=0,
            role="reference, already inspected"
            if seed == 21
            else "primary independent dictionary start",
            seconds_to_freeze=time.monotonic() - start,
        )
        s.save("frozen_nomination", nomination)
        s.write("development/nomination.json", nomination)
        s.write(
            "development/comparison.json",
            dict(
                control=control,
                edited=candidate,
                deltas=accum.category_deltas(candidate, control),
                gate=gate,
            ),
        )
        analysis = mature.reanalyze(arm)
        s.write(
            "split_roles.json",
            dict(
                ids={k: p["ids"] for k, p in arm["packs"].items()},
                targets=strong["spec"]["targets"],
                new_test_used=False,
                train="Common GGF continuation, refreshed profiles/dependencies and cumulative exchange checks.",
                select="Historical development: one fixed batch check; no best-prefix or second nominee.",
                test="All roots and all candidates fixed before new test extraction. No candidate retry.",
            ),
        )
        s.save("seed_frozen", nomination)
        rows.append(nomination)
        master.event(
            "seed_frozen",
            **nomination,
            post_graph_candidates=len(analysis["next_proposals"]),
        )
    result = dict(
        complete=True,
        models=rows,
        primary_seeds=[22, 23],
        reference_seed=21,
        fresh_test_used=False,
        new_maturity_updates=sum(r["maturity_new_updates"] for r in rows),
        edit_gradient_updates=0,
    )
    master.save("all_models_frozen", result)
    master.write("all_models_frozen.json", result)
    memory.assert_immutable(ctx["stamps"])
    return result


def extract_fresh(ctx, coco, token_callback):
    """Shard resume and normalized mmap; no whole-test activation concatenation."""
    s = ctx["store"]
    frozen = memory.required(s, "all_models_frozen")
    if not frozen["complete"]:
        raise RuntimeError("Freeze all candidates before extraction")
    plan = memory.required(s, "fresh_plan")
    ids = plan["splits"]["test"]
    cfg = ctx["study_cfg"]
    path = ctx["local"] / "fresh_normalized.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(len(ids), 196, 768)
    )
    processor = model = None
    masks = []
    cats = sorted(coco.cats)
    try:
        for lo in range(0, len(ids), cfg["extract_chunk"]):
            chosen = ids[lo : lo + cfg["extract_chunk"]]
            key = f"features_test_{lo:06d}"
            part = s.load(key)
            if part is None:
                if model is None:
                    processor, model = dataset.backbone(
                        ctx["provenance"], ctx["device"], token_callback
                    )
                    verified = getattr(model, "_agf_backbone_verification", None)
                    if verified is not None:
                        s.write("fresh_backbone_verification.json", verified)
                tensors = []
                mm = []
                geometries = []
                hashes = []
                for b in range(0, len(chosen), cfg["extract_batch"]):
                    images = []
                    for iid in chosen[b : b + cfg["extract_batch"]]:
                        im = coco.imgs[iid]
                        file = data.download(
                            data.COCO_BUCKET + "train2017/" + im["file_name"],
                            ctx["local"] / "images" / im["file_name"],
                        )
                        g = data.geometry(im["width"], im["height"])
                        with Image.open(file) as image:
                            if image.size != (im["width"], im["height"]):
                                raise ValueError("Image/mask geometry mismatch")
                            images.append(data.transform_image(image.convert("RGB"), g))
                        geometries.append(g)
                        hashes.append(sha_file(file))
                        mm.append(data.masks_for_image(coco, iid, g, cats))
                    xx = processor(
                        images=images,
                        do_resize=False,
                        do_center_crop=False,
                        return_tensors="pt",
                    ).to(ctx["device"])
                    with torch.inference_mode():
                        features = model(**xx).last_hidden_state[:, 5:].float().cpu()
                    if tuple(features.shape[1:]) != (196, 768) or not bool(
                        torch.isfinite(features).all()
                    ):
                        raise ValueError("Invalid DINO patch extraction")
                    tensors.append(features.half())
                    del xx, features
                part = dict(
                    ids=chosen,
                    x=torch.cat(tensors),
                    masks=mm,
                    geometry=geometries,
                    image_hashes=hashes,
                )
                s.save(key, part)
                del tensors
            if part["ids"] != chosen:
                raise ValueError("Fresh extraction shard identity changed")
            x[lo : lo + len(chosen)] = data.normalize(
                part["x"].float().numpy(), ctx["norm"]
            )
            x.flush()
            masks.extend(part["masks"])
            del part
            gc.collect()
            s.event(
                "extraction", split="new_test", done=lo + len(chosen), total=len(ids)
            )
        pack = dict(ids=ids, x=x, masks=masks)
        observed = coverage.evidence(pack, ctx["eval_cfg"]).all(2).sum(0)
        expected = np.asarray(plan["coverage"]["split_counts"]["test"])
        if not np.array_equal(observed, expected):
            raise ValueError("Actual fresh mask coverage differs from plan")
        s.write(
            "fresh_activation_identity.json",
            dict(
                ids=ids,
                normalized_sha=memory.streaming_array_sha(x),
                shape=list(x.shape),
                normalization_source="Original training normalization; never refit on fresh test",
                backbone=ctx["provenance"]["backbone"],
                coverage=observed.tolist(),
            ),
        )
        return pack
    finally:
        del processor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def verify_frozen(s, row):
    root = memory.required(s, row["root"])
    edit = memory.required(s, row["candidate"])
    if (
        accum.weight_sha(root) != row["root_sha"]
        or accum.weight_sha(edit) != row["candidate_sha"]
    ):
        raise ValueError("A frozen model changed after nomination")
    return root, edit


def fresh_pair(arm, edit, root):
    s = arm["store"]
    saved = s.load("fresh_paired")
    if saved is not None:
        return saved
    mapping = dict(
        mappings=[
            dict(
                category_index=t["category_index"],
                detector=dict(group=t["detector_group"]),
                contrast=dict(groups=[t["contrast_group"]]),
            )
            for t in root["spec"]["targets"]
        ]
    )

    def compact(snap, label):
        m = prior.model(arm, snap)
        p = arm["packs"]["fresh"]
        a = metrics.audit_arrays(
            m, p["x"], p["masks"], p["ids"], arm["cfg"], arm["device"]
        )
        result = study.compact_audit(a, mapping, arm["cfg"])
        del m, a
        gc.collect()
        folder = s.results / "image_audits"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / ("fresh_" + label + ".npz"),
            **{k: v for k, v in result.items() if isinstance(v, np.ndarray)},
            image_ids=np.array(result["image_ids"]),
            category_indices=np.array(result["category_indices"]),
            annotation_ids=np.array(result["objects"]),
        )
        return result

    a = compact(edit, "edited")
    b = compact(root, "control")
    ci = study.paired(a, b, dict(arm["cfg"], bootstrap=200, fresh_test_seed=106051))
    ci["scope"] = (
        "Paired whole-image interval on the newly allocated test set, conditional on one trained start. Same images across starts; not independent datasets."
    )
    s.save("fresh_paired", ci)
    s.write("paired/fresh.json", ci)
    return ci


def aggregate(rows):
    primary = [r for r in rows if r["seed"] in (22, 23)]
    if len(primary) != 2 or len({r["seed"] for r in primary}) != 2:
        raise ValueError("Cannot omit a failed primary start")
    return dict(
        primary_seeds=[22, 23],
        primary_count=2,
        primary_capture_gain_mean=float(np.mean([r["capture_gain"] for r in primary])),
        primary_ap_gain_mean=float(np.mean([r["ap_gain"] for r in primary])),
        primary_seeds_passing_all_guards=sum(
            r["fresh_gate"]["passed"] and r["deploy_candidate"] for r in primary
        ),
        primary_seeds_positive_capture=sum(r["capture_gain"] > 0 for r in primary),
        primary_seeds_positive_capture_and_ap=sum(
            r["capture_gain"] > 0 and r["ap_gain"] > 0 for r in primary
        ),
        repeatable_guarded_advantage=all(
            r["fresh_gate"]["passed"] and r["deploy_candidate"] for r in primary
        ),
        rule="Both predeclared primary starts must have positive mean capture and pass unchanged preservation checks. Report point estimates and intervals, not only a Boolean. Seed 21 is secondary.",
    )


def evaluate(ctx, test):
    master = ctx["store"]
    frozen = memory.required(master, "all_models_frozen")
    rows = []
    for item in frozen["models"]:
        seed = item["seed"]
        arm = ctx["arms"][seed]
        s = arm["store"]
        root, edit = verify_frozen(s, item)
        arm["initial"] = root
        arm["source_weights_sha"] = item["root_sha"]
        arm["packs"] = dict(ctx["packs"], fresh=test)
        saved = s.load("fresh_result")
        if saved is not None:
            rows.append(saved)
            continue
        s.event("fresh_scoring_start", seed=seed, images=len(test["ids"]))
        control = accum.measured(arm, "fresh_control", root, "fresh")
        candidate = accum.measured(arm, "fresh_candidate", edit, "fresh")
        gate = accum.positive_gate(candidate, control, control, arm["protocol"])
        ci = fresh_pair(arm, edit, root)
        row = dict(
            **item,
            control=control,
            edited=candidate,
            fresh_gate=gate,
            paired=ci,
            capture_gain=candidate["macro_capture"] - control["macro_capture"],
            ap_gain=candidate["macro_ap"] - control["macro_ap"],
            ve_gain=candidate["reconstruction"]["ve"] - control["reconstruction"]["ve"],
            deltas=accum.category_deltas(candidate, control),
            deployed_arm="AGF" if item["deploy_candidate"] else "GGF fallback",
            images=len(test["ids"]),
            test_used=True,
            no_test_reselection=True,
        )
        s.save("fresh_result", row)
        s.write("fresh_result.json", row)
        rows.append(row)
        master.event(
            "fresh_seed_complete",
            seed=seed,
            capture_gain_pp=100 * row["capture_gain"],
            ap_gain_pp=100 * row["ap_gain"],
            ve_gain_pp=100 * row["ve_gain"],
            passed=gate["passed"],
            failures=gate["failures"],
            development_adopted=item["deploy_candidate"],
        )
    result = dict(
        complete=True,
        results=rows,
        **aggregate(rows),
        new_test_images=len(test["ids"]),
        new_maturity_updates=frozen["new_maturity_updates"],
        edit_gradient_updates=0,
        scope=ctx["study_cfg"]["scope"],
        all_seed_results_included=True,
        limits="Two primary starts, one shared coverage-enriched image set. Fixed width 3 and Top-16. Supervised object proxies, not exact concept recovery or a direct feature-splitting rate. No downstream task test or wall-time-matched advantage claimed.",
    )
    master.save("experiment_complete", result)
    master.write("summary.json", result)
    charts(ctx, rows)
    memory.assert_immutable(ctx["stamps"])
    return result


def charts(ctx, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = ctx["store"]
    folder = s.results / "figures"
    folder.mkdir(exist_ok=True)
    tab = []
    for row in rows:
        for name, report in [
            ("Mature GGF", row["control"]),
            ("Frozen AGF candidate", row["edited"]),
        ]:
            tab.append(
                dict(
                    seed=row["seed"], arm=name, role=row["role"], **bank.scalar(report)
                )
            )
    with (s.results / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, metric, title in zip(
        axes,
        ("capture_gain", "ap_gain", "ve_gain"),
        ("Capture gain", "Detection AP gain", "Reconstruction VE change"),
    ):
        ax.bar(
            range(3),
            [100 * r[metric] for r in rows],
            color=["#a6b5bf", "#008e80", "#008e80"],
        )
        ax.set_xticks(range(3), ["21 (reference)", "22 (primary)", "23 (primary)"])
        ax.set_ylabel("percentage points")
        ax.set_title(title)
        ax.axhline(0, color="#444", lw=0.7)
    fig.suptitle("Fresh images: all saved starts, including failures")
    fig.tight_layout()
    fig.savefig(folder / "fresh_gains.png", dpi=160)
    plt.close(fig)


def export(ctx):
    s = ctx["store"]
    path = s.output / f"agf_independent_regroup_{s.run_id[:12]}_results.zip"
    hashes = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        roots = [("results", s.results)] + [
            (f"seeds/{seed}", a["store"].results) for seed, a in ctx["arms"].items()
        ]
        for prefix, root in roots:
            for p in sorted(root.rglob("*")):
                if p.is_file():
                    name = prefix + "/" + p.relative_to(root).as_posix()
                    z.write(p, name)
                    hashes[name] = sha_file(p)
        z.writestr("artifact_hashes.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise OSError("Export corruption")
    return path
