"""Three native starts, matched learning controls, frozen final object audit."""

import copy
import csv
import importlib.metadata
import json
import shutil
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import torch

from coco_baselines import metrics, training
from coco_baselines.storage import identity, serial
from . import core, inputs, protection, coverage

EXPECTED_PARENT = "aa65d94cee15de924d80f9b3dbd100167f63d39322221387de40cd46aaf2ef39"
DEFAULT = dict(
    version=3,
    coverage_seed=61201,
    coverage_max_extra_images=128,
    methods=["vanilla", "grassmannian", "group_lasso"],
    device="cuda",
    lr=3e-4,
    snr=0.1,
    warm_updates=600,
    rounds=2,
    refit_updates=200,
    candidates_per_round=4,
    equal_search_compute_control=True,
    images_per_update=8,
    batch=2048,
    evaluate_every=100,
    local_save_every=20,
    grad_clip=5.0,
    lambda_capture=0.05,
    lambda_detection=0.01,
    score_ap_weight=0.2,
    fit_reconstruction_tolerance=0.02,
    accept_reconstruction_tolerance=0.01,
    global_reconstruction_tolerance=0.02,
    min_score_gain=0.002,
    macro_tradeoff_tolerance=0.005,
    category_preservation_tolerance=0.03,
    anchor_ablation=True,
    protect_capture_threshold=0.5,
    protect_ap_threshold=0.8,
    protected_metric_tolerance=0.03,
    lambda_contribution_anchor=0.05,
    lambda_gram_anchor=0.01,
    anchor_energy_floor=1e-8,
    anchor_seed=61104,
    anchor_examples_per_group=128,
    anchor_patches_per_image=8,
    anchor_pairs_per_update=2,
    diagnostic_grad_every=100,
    target_min_images=10,
    decision_min_images=3,
    proposal_images=200,
    proposal_group_pool=4,
    fresh_test_images=600,
    fresh_test_seed=61101,
    partition_seed=61102,
    training_seed=61103,
    patch_sample_seed=61002,
    audit_image_batch=8,
    detection_patches=32,
    preview_images=6,
    positive_fraction=0.5,
    negative_fraction=0.1,
    min_object_patch_mass=2.0,
    min_background_patches=4,
    min_category_images=5,
    max_audit_groups=4,
    capture_threshold=0.8,
    bootstrap=100,
    extract_chunk=20,
    extract_batch=8,
)


def config(preset="pilot"):
    c = copy.deepcopy(DEFAULT)
    c["preset"] = preset
    if preset == "smoke":
        c.update(
            warm_updates=10,
            rounds=1,
            refit_updates=10,
            evaluate_every=5,
            local_save_every=5,
            candidates_per_round=2,
            proposal_images=32,
            bootstrap=5,
            fresh_test_images=40,
            target_min_images=2,
            min_category_images=2,
            decision_min_images=1,
            diagnostic_grad_every=5,
            anchor_examples_per_group=16,
        )
    elif preset != "pilot":
        raise ValueError("Choose pilot or smoke; smoke is plumbing only")
    return c


def build_store(
    output, local, cfg, parent, source_sha, require_drive=True, previous_root=None
):
    versions = {
        p: importlib.metadata.version(p)
        for p in (
            "torch",
            "numpy",
            "scikit-learn",
            "transformers",
            "pycocotools",
            "einops",
        )
    }
    store = inputs.CompatibleStore(
        output,
        dict(
            config=cfg,
            parent_run_id=parent.run_id,
            source_sha=source_sha,
            numerical_versions=versions,
        ),
        local,
        require_drive,
    )
    store.attach_previous(previous_root)
    store.write(
        "environment.json",
        dict(
            current=versions,
            parent=parent.config["numerical_versions"],
            policy="Reuse verified saved activations and weights; changed environment has a distinct output identity. No forced Torch reinstall.",
        ),
    )
    store.write(
        "thresholds.json",
        dict(
            values={
                k: cfg[k]
                for k in cfg
                if "tolerance" in k
                or k.startswith("lambda")
                or k.startswith("min_score")
            },
            status="Predeclared exploratory engineering settings, not calibrated optimal thresholds or a proof of acceptance reliability. Repeated validation/confirmation reuse makes this a development study; final fresh test is never used for decisions.",
        ),
    )
    return store


def small_training_pack(packs, cfg):
    n = min(cfg["proposal_images"], len(packs["train"]["ids"]))
    take = np.random.default_rng(cfg["training_seed"]).permutation(
        len(packs["train"]["ids"])
    )[:n]
    return inputs.split_pack(packs["train"], take)


def assess(weights, method, base_cfg, pack, targets, cfg, protection_pack=None):
    m = core.new_model(method, base_cfg, weights, cfg["device"])
    v = core.validation(
        m, pack, targets, cfg, cfg["device"], protection_pack=protection_pack
    )
    del m
    return v


def prepare_protection(store, source, choices, packs, provenance, cfg):
    """Every native model must pass against itself BEFORE any adaptation fits."""
    base = source.config["config"]
    items = []
    for choice in choices:
        method = choice["method"]
        tag = method + "_s" + str(choice["seed"])
        state = source.load(choice["key"])
        native = state["best"] if choice["sparsity_matched"] else state["fallback"]
        audit = train_audit(
            store, tag + "_training_audit", native, method, base, packs["train"], cfg
        )
        targets = store.load(tag + "_targets")
        if targets is None:
            targets = core.choose_targets(audit, cfg, provenance["categories"])
            store.save(tag + "_targets", targets)
        items.append((method, tag, native, targets))
    report = {}
    try:
        if "historical_development" in packs:
            a, b, report = coverage.plan(
                packs["select"],
                packs["confirm"],
                packs["historical_development"],
                [x[3] for x in items],
                cfg,
            )
            packs["protection_select"] = a
            packs["protection_confirm"] = b
        else:
            # Small local fixtures and explicitly covered existing sets need no pool.
            packs["protection_select"] = packs["select"]
            packs["protection_confirm"] = packs["confirm"]
            report = dict(
                passed=True,
                supplement_ids={},
                source="Existing covered development sets",
            )
        forbidden = set(packs["train"]["ids"]) | set(
            packs.get("mapping", {}).get("ids", [])
        )
        if any(
            forbidden & set(packs[k]["ids"])
            for k in ("protection_select", "protection_confirm")
        ):
            raise ValueError("Protection data leaks training or final mapping")
        checks = []
        for method, tag, native, targets in items:
            a = assess(native, method, base, packs["protection_select"], targets, cfg)
            b = assess(native, method, base, packs["protection_confirm"], targets, cfg)
            m = core.new_model(method, base, native, cfg["device"])
            book = protection.build(m, packs["train"], targets, cfg, a, b)
            del m
            book["preservation_image_ids"] = dict(
                selection=packs["protection_select"]["ids"],
                confirmation=packs["protection_confirm"]["ids"],
            )
            book["id"] = identity(
                dict(
                    native_anchor_id=book["id"],
                    preservation_image_ids=book["preservation_image_ids"],
                )
            )
            ga = protection.guard(a, book, cfg)
            gb = protection.guard(b, book, cfg, "confirmation")
            checks.append(dict(method=method, selection=ga, confirmation=gb))
            store.save(tag + "_anchors", book)
            if not ga["passed"] or not gb["passed"]:
                raise ValueError(
                    "Native preservation self-check failed before training: "
                    + method
                    + " "
                    + str(ga["reasons"] + gb["reasons"])
                )
        report.update(native_self_checks=checks, passed=True)
        store.write("coverage_preflight.json", report)
        store.event(
            "preservation_coverage_preflight_passed",
            methods=len(items),
            supplement_counts={k: len(v) for k, v in report["supplement_ids"].items()},
        )
    except BaseException as exc:
        report.update(
            passed=False,
            status="setup_coverage_error",
            error=str(exc),
            training_started=False,
        )
        store.write("coverage_preflight.json", report)
        raise
    packs.pop("historical_development", None)
    return report


def train_audit(store, key, w, method, base_cfg, pack, cfg):
    old = store.load(key)
    if old is not None:
        if old["weights_sha"] != training.weight_sha(w):
            raise ValueError("Proposal audit stale")
        return old["audit"]
    m = core.new_model(method, base_cfg, w, cfg["device"])
    a = metrics.audit_arrays(
        m, pack["x"], pack["masks"], pack["ids"], cfg, cfg["device"]
    )
    del m
    store.save(key, dict(weights_sha=training.weight_sha(w), audit=a))
    return a


def train_all(store, source, choices, packs, provenance, cfg):
    done = store.load("all_models_frozen")
    if done is not None:
        return done
    if "protection_select" not in packs:
        prepare_protection(store, source, choices, packs, provenance, cfg)
    base = source.config["config"]
    proposal_pack = small_training_pack(packs, cfg)
    final = []
    for choice in choices:
        method = choice["method"]
        tag = method + "_s" + str(choice["seed"])
        state = source.load(choice["key"])
        native = state["best"] if choice["sparsity_matched"] else state["fallback"]
        audit = train_audit(
            store, tag + "_training_audit", native, method, base, packs["train"], cfg
        )
        targets = store.load(tag + "_targets")
        if targets is None:
            targets = core.choose_targets(audit, cfg, provenance["categories"])
            store.save(tag + "_targets", targets)
        store.write(
            "targets/" + tag + ".json",
            dict(
                targets=targets,
                label_source="Training annotations only; detector and contrast groups can differ.",
            ),
        )
        start = assess(native, method, base, packs["select"], targets, cfg)
        confirm_start = assess(native, method, base, packs["confirm"], targets, cfg)
        book = store.load(tag + "_anchors")
        if book is None:
            m = core.new_model(method, base, native, cfg["device"])
            book = protection.build(
                m, packs["train"], targets, cfg, start, confirm_start
            )
            del m
            store.save(tag + "_anchors", book)
        store.write("protection/" + tag + ".json", protection.summary(book))
        store.event(
            "protection_prepared",
            method=method,
            **{
                k: v
                for k, v in protection.summary(book).items()
                if k not in ("entries", "skipped", "source", "promotion_policy")
            },
        )
        seed = cfg["training_seed"] + choice["seed"]
        horizon = cfg["warm_updates"] + cfg["rounds"] * cfg["refit_updates"]
        store.save(tag + "_native", dict(best=native, updates=0))
        arms = [
            dict(
                arm="native",
                key=tag + "_native",
                method=method,
                seed=choice["seed"],
                updates=0,
                weights_sha=training.weight_sha(native),
            )
        ]
        w = native
        for segment, count in enumerate(
            [cfg["warm_updates"]] + [cfg["refit_updates"]] * cfg["rounds"]
        ):
            key = f"{tag}_continued_reconstruction_seg{segment}"
            st = core.fit(
                store,
                key,
                method,
                base,
                w,
                packs["train"],
                packs["select"],
                targets,
                cfg,
                count,
                False,
                seed + segment,
                cfg["device"],
            )
            w = st["best"]
        arms.append(
            dict(
                arm="continued_reconstruction",
                key=key,
                method=method,
                seed=choice["seed"],
                updates=horizon,
                weights_sha=training.weight_sha(w),
            )
        )
        requested = cfg.get("adaptation_variants")
        if requested is None:
            variants = [("anchored", book)]
            if cfg["anchor_ablation"]:
                variants.insert(0, ("unanchored", None))
        else:
            if (
                not requested
                or len(set(requested)) != len(requested)
                or set(requested) - {"unanchored", "anchored"}
            ):
                raise ValueError(
                    "Choose unique anchored/unanchored adaptation variants"
                )
            variants = [
                (name, book if name == "anchored" else None) for name in requested
            ]
        for variant, anchors in variants:
            protect_pack = packs["protection_select"] if anchors is not None else None

            def measure(w, split):
                extra = packs["protection_" + split] if anchors is not None else None
                return assess(w, method, base, packs[split], targets, cfg, extra)

            fixed_arm = (
                "supervised_fixed" if anchors is None else "anchored_supervised_fixed"
            )
            adaptive_arm = (
                "supervised_adaptive"
                if anchors is None
                else "anchored_supervised_adaptive"
            )
            compute_arm = (
                "supervised_fixed_equal_search_compute"
                if anchors is None
                else "anchored_supervised_fixed_equal_search_compute"
            )
            prefix = tag + "_" + variant
            w = native
            for segment, count in enumerate(
                [cfg["warm_updates"]] + [cfg["refit_updates"]] * cfg["rounds"]
            ):
                key = f"{tag}_{fixed_arm}_seg{segment}"
                st = core.fit(
                    store,
                    key,
                    method,
                    base,
                    w,
                    packs["train"],
                    packs["select"],
                    targets,
                    cfg,
                    count,
                    True,
                    seed + segment,
                    cfg["device"],
                    anchors=anchors,
                    protection_pack=protect_pack,
                )
                w = st["best"]
            arms.append(
                dict(
                    arm=fixed_arm,
                    key=key,
                    method=method,
                    seed=choice["seed"],
                    updates=horizon,
                    weights_sha=training.weight_sha(w),
                )
            )
            warm = store.load(f"{tag}_{fixed_arm}_seg0")
            current = warm["best"]
            accepted = 0
            search_updates = cfg["warm_updates"]
            if anchors is not None:
                warm_selection = protection.guard(
                    measure(current, "select"), anchors, cfg
                )
                warm_confirmation = protection.guard(
                    measure(current, "confirm"), anchors, cfg, "confirmation"
                )
                warm_ok = warm_selection["passed"] and warm_confirmation["passed"]
                if not warm_ok:
                    current = native
                store.write(
                    "protection/" + prefix + "_warmup.json",
                    dict(
                        selection=warm_selection,
                        confirmation=warm_confirmation,
                        kept_native=not warm_ok,
                    ),
                )
                store.event(
                    "warmup_protection_check",
                    method=method,
                    variant=variant,
                    kept_native=not warm_ok,
                )
            for r in range(cfg["rounds"]):
                round_key = f"{prefix}_round{r}"
                old = store.load(round_key)
                if old is not None:
                    current = old["weights"]
                    accepted += int(old["accepted"])
                    search_updates += old["gradient_updates"]
                    print("Reused", round_key, flush=True)
                    continue
                audit = train_audit(
                    store,
                    round_key + "_proposal_audit",
                    current,
                    method,
                    base,
                    proposal_pack,
                    cfg,
                )
                props = store.load(round_key + "_proposals")
                if props is None:
                    props = core.proposals(audit, targets, cfg)
                    for p in props:
                        p["protected_pairs_touched"] = [
                            dict(
                                category=e["category"], role=e["role"], group=e["group"]
                            )
                            for e in book["entries"]
                            if e["group"] in p["pair"]
                        ]
                    store.save(round_key + "_proposals", props)
                store.write("proposals/" + round_key + ".json", props)
                store.event(
                    "proposal_screen",
                    method=method,
                    variant=variant,
                    round=r + 1,
                    candidates=len(props),
                    positive_pair_gains=sum(p["training_pair_gain"] > 0 for p in props),
                    empty_protection=not bool(book["entries"]),
                )
                key = round_key + "_control"
                control = core.fit(
                    store,
                    key,
                    method,
                    base,
                    current,
                    packs["train"],
                    packs["select"],
                    targets,
                    cfg,
                    cfg["refit_updates"],
                    True,
                    seed + r + 1,
                    cfg["device"],
                    anchors=anchors,
                    protection_pack=protect_pack,
                )
                control_val = measure(control["best"], "select")
                control_confirm = measure(control["best"], "confirm")
                control_guard = protection.guard(control_val, anchors, cfg)
                control_confirm_guard = protection.guard(
                    control_confirm, anchors, cfg, "confirmation"
                )
                before = measure(current, "select")
                candidates = []
                for j, p in enumerate(props):
                    key = round_key + f"_candidate{j}"
                    edited = core.transform_weights(
                        method, current, p["pair"], p["Q"], base["block_rank"]
                    )
                    immediate = measure(edited, "select")
                    immediate_guard = protection.guard(immediate, anchors, cfg)
                    store.event(
                        "edit_proposed",
                        key=key,
                        kind=p["kind"],
                        pair=p["pair"],
                        category=p["category"],
                        initial_mse=before["reconstruction"]["mse"],
                        edited_mse=immediate["reconstruction"]["mse"],
                        immediate_mse_relative_change=immediate["reconstruction"]["mse"]
                        / before["reconstruction"]["mse"]
                        - 1,
                        immediate_protection=immediate_guard,
                    )
                    st = core.fit(
                        store,
                        key,
                        method,
                        base,
                        edited,
                        packs["train"],
                        packs["select"],
                        targets,
                        cfg,
                        cfg["refit_updates"],
                        True,
                        seed + r + 1,
                        cfg["device"],
                        anchors=anchors,
                        protection_pack=protect_pack,
                    )
                    value = measure(st["best"], "select")
                    decision = core.compare(
                        value,
                        control_val,
                        start,
                        cfg,
                        base,
                        method,
                        p["category_index"],
                        anchors,
                    )
                    diagnostic = dict(
                        key=key,
                        proposal=p,
                        immediate=immediate,
                        before=before,
                        after=value,
                        immediate_protection=immediate_guard,
                        selection_decision=decision,
                        best_update=st["best_update"],
                        valid_checkpoint_found=st["best_valid"],
                        recovery_mse_delta=value["reconstruction"]["mse"]
                        - immediate["reconstruction"]["mse"],
                        observational_failure_components=failure_components(
                            st, decision, p
                        ),
                    )
                    store.write("decisions/" + key + ".json", diagnostic)
                    candidates.append(
                        dict(
                            key=key,
                            score=value["score"],
                            selection_pass=decision["accepted"],
                            proposal=p,
                            diagnostic=diagnostic,
                        )
                    )
                eligible = [x for x in candidates if x["selection_pass"]]
                chosen = max(eligible, key=lambda x: x["score"]) if eligible else None
                confirm = None
                accept = False
                if chosen:
                    selected = store.load(chosen["key"])
                    candidate_confirm = measure(selected["best"], "confirm")
                    confirm = core.compare(
                        candidate_confirm,
                        control_confirm,
                        confirm_start,
                        cfg,
                        base,
                        method,
                        chosen["proposal"]["category_index"],
                        anchors,
                        "confirmation",
                    )
                    accept = confirm["accepted"]
                # A no-edit fallback must also preserve the ORIGINAL successes;
                # otherwise keep the previous accepted weights, not a degraded control.
                fallback_ok = (
                    control_guard["passed"] and control_confirm_guard["passed"]
                )
                current = (
                    store.load(chosen["key"])["best"]
                    if accept
                    else (control["best"] if fallback_ok else current)
                )
                count = (len(props) + 1) * cfg["refit_updates"]
                search_updates += count
                accepted += int(accept)
                record = dict(
                    weights=current,
                    accepted=accept,
                    selected_candidate=None if chosen is None else chosen["key"],
                    confirmation=confirm,
                    control_protection=control_guard,
                    control_confirmation_protection=control_confirm_guard,
                    fallback_kept_previous=not accept and not fallback_ok,
                    gradient_updates=count,
                    candidates=[
                        {k: v for k, v in x.items() if k != "diagnostic"}
                        for x in candidates
                    ],
                    no_proposals=not bool(props),
                    no_selection_candidate_passed=not bool(eligible),
                    confirmation_reused_across_rounds=True,
                )
                store.save(round_key, record)
                store.write(
                    "decisions/" + round_key + ".json",
                    {k: v for k, v in record.items() if k != "weights"},
                )
                store.event(
                    "structural_round_complete",
                    method=method,
                    variant=variant,
                    round=r + 1,
                    accepted=accept,
                    accepted_so_far=accepted,
                    fallback_kept_previous=record["fallback_kept_previous"],
                    rejection_reasons=["no_candidate_passed_selection"]
                    if chosen is None
                    else confirm["reasons"],
                )
            key = prefix + "_adaptive"
            store.save(
                key,
                dict(
                    best=current,
                    updates=horizon,
                    search_gradient_updates=search_updates,
                    accepted_edits=accepted,
                ),
            )
            arms.append(
                dict(
                    arm=adaptive_arm,
                    key=key,
                    method=method,
                    seed=choice["seed"],
                    updates=horizon,
                    search_gradient_updates=search_updates,
                    accepted_edits=accepted,
                    protected_pairs=0 if anchors is None else len(anchors["entries"]),
                    weights_sha=training.weight_sha(current),
                )
            )
            if cfg["equal_search_compute_control"]:
                w = warm["best"]
                extra = search_updates - cfg["warm_updates"]
                segment = 0
                while extra > 0:
                    count = min(cfg["refit_updates"], extra)
                    key = f"{tag}_{compute_arm}_seg{segment}"
                    st = core.fit(
                        store,
                        key,
                        method,
                        base,
                        w,
                        packs["train"],
                        packs["select"],
                        targets,
                        cfg,
                        count,
                        True,
                        seed + segment + 1,
                        cfg["device"],
                        anchors=anchors,
                        protection_pack=protect_pack,
                    )
                    w = st["best"]
                    extra -= count
                    segment += 1
                arms.append(
                    dict(
                        arm=compute_arm,
                        key=key,
                        method=method,
                        seed=choice["seed"],
                        updates=search_updates,
                        weights_sha=training.weight_sha(w),
                    )
                )
        final.extend(arms)
    store.save("all_models_frozen", final)
    store.write(
        "all_models_frozen.json",
        dict(
            models=final,
            test_annotations_used=False,
            supervised=True,
            novelty_test="Structural adaptation versus fixed-boundary supervision at equal search update budget; anchored and unanchored variants are recorded separately.",
        ),
    )
    return final


def failure_components(state, decision, proposal):
    rows = state["history"][1:]
    out = []
    if proposal["training_pair_gain"] <= 0:
        out.append("proposal:no_positive_training_complementarity")
    if not state["best_valid"]:
        out.append("refit:no_checkpoint_meets_protection_and_reconstruction")
    if rows and all(r["objects_in_batch"] == 0 for r in rows):
        out.append("supervision:no_object_contrast_examples_in_evaluated_batches")
    if rows and all(r["decoder_gradient_norm"] == 0 for r in rows):
        out.append("decoder:zero_observed_gradient")
    if rows and all(r["encoder_gradient_norm"] == 0 for r in rows):
        out.append("encoder:zero_observed_gradient")
    for reason in decision["reasons"]:
        out.append("selection:" + reason)
    return dict(
        observations=out,
        limitation="Logs identify measured failure conditions, not guaranteed root causes or globally optimal remedies.",
    )


def evaluate_all(store, source, arms, packs, fresh, cfg, provenance):
    base_cfg = source.config["config"]
    reports = []
    for arm in arms:
        key = arm["key"]
        old = store.load("report_" + key)
        if old is not None:
            reports.append(old)
            continue
        st = store.load(key)
        if training.weight_sha(st["best"]) != arm["weights_sha"]:
            raise ValueError("Final arm changed")
        m = core.new_model(arm["method"], base_cfg, st["best"], cfg["device"])
        frozen = store.load("mapping_" + key)
        if frozen is None:
            a = metrics.audit_arrays(
                m,
                packs["mapping"]["x"],
                packs["mapping"]["masks"],
                packs["mapping"]["ids"],
                cfg,
                cfg["device"],
            )
            frozen = metrics.calibrate(a, cfg, provenance["categories"])
            store.save("mapping_" + key, frozen)
            store.write(
                "mappings/" + key + ".json",
                dict(
                    mappings=frozen["mappings"],
                    weights_sha=arm["weights_sha"],
                    scope="Disjoint final mapping calibration; not used for structural acceptance.",
                ),
            )
            del a
        audit_key = "audit_fresh_" + key
        a = store.load(audit_key)
        if a is None:
            a = metrics.audit_arrays(
                m, fresh["x"], fresh["masks"], fresh["ids"], cfg, cfg["device"]
            )
            store.save(audit_key, a)
        rows = metrics.evaluate(a, frozen, cfg, cfg["fresh_test_seed"])
        recon = training.measure(
            m, fresh["x"].reshape(-1, m.d), cfg["device"], cfg["batch"]
        )
        det = [r["detection"] for r in rows if r["detection"] is not None]
        con = [r["contrast"] for r in rows if r["contrast"] is not None]
        mean = lambda values: float(np.mean(values)) if len(values) else None
        summary = dict(
            **arm,
            reconstruction=recon,
            mean_single_group_ap=mean([r["ap"] for r in det]),
            mean_multi_group_probe_ap=mean([r["multi_group_probe_ap"] for r in det]),
            mean_single_group_contrast_capture=mean(
                [r["single_group_capture"] for r in con]
            ),
            mean_multi_group_contrast_capture=mean(
                [r["multi_group_capture"] for r in con]
            ),
            mean_all_group_contrast_capture=mean([r["all_group_capture"] for r in con]),
            detection_categories=len(det),
            contrast_categories=len(con),
            contrast_spread_categories=sum(
                r["distributed_object_contrast"] for r in con
            ),
            limitation="Supervised object-grounded proxies; no exact manifold recovery or feature splitting rate. One pretrained seed, bounded search, fresh COCO image holdout.",
        )
        report = dict(summary=summary, categories=rows, skipped_objects=a["skipped"])
        store.save("report_" + key, report)
        store.write("reports/" + key + ".json", report)
        reports.append(report)
        store.event(
            "fresh_audit_complete",
            method=arm["method"],
            arm=arm["arm"],
            summary=summary,
        )
        del m, a
        if cfg["device"].startswith("cuda"):
            torch.cuda.empty_cache()
    store.write(
        "summary.json",
        dict(
            reports=[r["summary"] for r in reports],
            scope="Object-supervised adaptation with disjoint new test images. Training and acceptance use human annotations; initial native BSF training was unsupervised.",
        ),
    )
    comparisons(store, reports, cfg)
    protection_report(store, reports, cfg)
    component_report(store)
    export_categories(store, reports)
    return reports


def comparisons(store, reports, cfg):
    groups = [
        (
            "supervised_adaptive",
            (
                "native",
                "continued_reconstruction",
                "supervised_fixed",
                "supervised_fixed_equal_search_compute",
            ),
        ),
        (
            "anchored_supervised_adaptive",
            (
                "native",
                "continued_reconstruction",
                "anchored_supervised_fixed",
                "anchored_supervised_fixed_equal_search_compute",
                "supervised_adaptive",
            ),
        ),
    ]
    table = []
    tests = []
    for method in cfg["methods"]:
        rr = {
            r["summary"]["arm"]: r for r in reports if r["summary"]["method"] == method
        }
        for adaptive, controls in groups:
            if adaptive not in rr:
                continue
            for other in controls:
                if other not in rr:
                    continue
                a = {r["category"]: r for r in rr[adaptive]["categories"]}
                b = {r["category"]: r for r in rr[other]["categories"]}
                for metric, field in (
                    ("detection", "ap"),
                    ("contrast", "single_group_capture"),
                ):
                    names = [c for c in a if c in b and a[c][metric] and b[c][metric]]
                    deltas = [a[c][metric][field] - b[c][metric][field] for c in names]
                    tests.append(
                        dict(
                            method=method,
                            adaptive_arm=adaptive,
                            comparison=other,
                            metric=field,
                            matched_categories=len(names),
                            mean_delta=float(np.mean(deltas)) if deltas else None,
                            category_wins=sum(d > 0 for d in deltas),
                            fresh_test=True,
                        )
                    )
                    for c, d in zip(names, deltas):
                        table.append(
                            dict(
                                method=method,
                                adaptive_arm=adaptive,
                                comparison=other,
                                category=c,
                                metric=field,
                                adaptive=a[c][metric][field],
                                control=b[c][metric][field],
                                delta=d,
                            )
                        )
                if other not in ("native", "continued_reconstruction"):
                    tests.append(
                        dict(
                            method=method,
                            adaptive_arm=adaptive,
                            comparison=other,
                            paired_intervals=paired_intervals(
                                store.load(
                                    "audit_fresh_" + rr[adaptive]["summary"]["key"]
                                ),
                                store.load(
                                    "audit_fresh_" + rr[other]["summary"]["key"]
                                ),
                                store.load("mapping_" + rr[adaptive]["summary"]["key"]),
                                store.load("mapping_" + rr[other]["summary"]["key"]),
                                cfg,
                            ),
                        )
                    )
    store.write("matched_comparisons.json", tests)
    if table:
        with (store.results / "per_category_comparisons.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=table[0].keys())
            writer.writeheader()
            writer.writerows(table)


def protection_report(store, reports, cfg):
    rows = []
    for method in cfg["methods"]:
        rr = [r["summary"] for r in reports if r["summary"]["method"] == method]
        native = next((r for r in rr if r["arm"] == "native"), None)
        if native is None:
            continue
        tag = method + "_s" + str(native["seed"])
        book = store.load(tag + "_anchors")
        targets = store.load(tag + "_targets")
        book = dict(
            book,
            fresh_reference=core.category_scores(
                store.load("audit_fresh_" + native["key"]), targets, cfg
            ),
        )
        for r in rr:
            original_scores = core.category_scores(
                store.load("audit_fresh_" + r["key"]), targets, cfg
            )
            check = protection.guard(original_scores, book, cfg, "fresh")
            rows.append(dict(method=method, arm=r["arm"], **check))
    store.write(
        "protected_group_fresh_audit.json",
        dict(
            rows=rows,
            note="Same original category/group IDs, compared against native on fresh images; audit only, never used to accept edits.",
        ),
    )


def component_report(store):
    cases = []
    for p in sorted((store.results / "decisions").glob("*candidate*.json")):
        r = json.loads(p.read_text())
        cases.append(
            dict(
                key=r["key"],
                proposal=r["proposal"],
                best_update=r["best_update"],
                reconstruction_spike=r["immediate"]["reconstruction"]["mse"]
                - r["before"]["reconstruction"]["mse"],
                reconstruction_recovery_delta=r["recovery_mse_delta"],
                **r["observational_failure_components"],
            )
        )
    rounds = []
    for p in sorted((store.results / "decisions").glob("*round*.json")):
        if "candidate" in p.name:
            continue
        r = json.loads(p.read_text())
        rounds.append(
            dict(
                key=p.stem,
                accepted=r["accepted"],
                selected_candidate=r["selected_candidate"],
                confirmation_reasons=None
                if r["confirmation"] is None
                else r["confirmation"]["reasons"],
                fallback_kept_previous=r["fallback_kept_previous"],
            )
        )
    store.write(
        "component_diagnostics.json",
        dict(
            candidates=cases,
            rounds=rounds,
            limitation="Evidence of specific failure conditions and gradient conflicts; not a guarantee of exact causal attribution.",
        ),
    )


def export_categories(store, reports):
    rows = []
    for report in reports:
        s = report["summary"]
        for c in report["categories"]:
            d, q = c["detection"] or {}, c["contrast"] or {}
            rows.append(
                dict(
                    method=s["method"],
                    arm=s["arm"],
                    seed=s["seed"],
                    category=c["category"],
                    calibration_positive_images=c["calibration_positive_images"],
                    test_positive_images=c["test_positive_images"],
                    one_group_ap=d.get("ap"),
                    multi_group_probe_ap=d.get("multi_group_probe_ap"),
                    one_group_f1=d.get("f1"),
                    iou=d.get("iou"),
                    chance_ap=d.get("prevalence"),
                    detector_group=d.get("group"),
                    contrast_group=q.get("single_group_id"),
                    same_group_selected=q.get("same_group_as_detector"),
                    contrast_group_object_ap=q.get("chosen_group_object_ap"),
                    detector_group_contrast_capture=q.get(
                        "detector_group_contrast_capture"
                    ),
                    one_group_contrast=q.get("single_group_capture"),
                    multi_group_contrast=q.get("multi_group_capture"),
                    all_group_contrast=q.get("all_group_capture"),
                    contrast_groups=q.get("groups"),
                    object_count=q.get("object_count"),
                    contrast_image_count=q.get("image_count"),
                    distributed_object_contrast=q.get("distributed_object_contrast"),
                    exclusions="; ".join(c["exclusions"]),
                )
            )
    if rows:
        local = store.rescue / "per_category.csv"
        with local.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        store.check_mount()
        shutil.copyfile(local, store.results / "per_category.csv")


def paired_intervals(a, b, fa, fb, cfg):
    """Whole-image paired bootstrap on frozen per-model mappings, macro common classes."""
    if a["image_ids"] != b["image_ids"]:
        raise ValueError("Paired audits use different images")
    ya, va = metrics.binary_labels(a, cfg)
    yb, vb = metrics.binary_labels(b, cfg)
    if not np.array_equal(ya, yb) or not np.array_equal(va, vb):
        raise ValueError("Paired masks differ")
    n = len(a["image_ids"])
    mappinga = fa["mappings"]
    mappingb = fb["mappings"]
    prepared = []
    oca = np.array([o["category"] for o in a["objects"]])
    ocb = np.array([o["category"] for o in b["objects"]])
    if [o["annotation_id"] for o in a["objects"]] != [
        o["annotation_id"] for o in b["objects"]
    ]:
        raise ValueError("Paired object alignment differs")
    for ma, mb in zip(mappinga, mappingb):
        c = ma["category_index"]
        item = dict(category_index=c)
        if ma["detector"] and mb["detector"]:
            p, q = metrics.image_counts(ya[:, c], va[:, c], a["image_index"])
            if min(p, q) >= cfg["min_category_images"]:
                item.update(
                    det_a=a["heat"][:, ma["detector"]["group"]],
                    det_b=b["heat"][:, mb["detector"]["group"]],
                )
        if ma["contrast"] and mb["contrast"]:
            ix = np.flatnonzero(oca == c)
            im = np.array([a["objects"][i]["image_index"] for i in ix])
            if len(np.unique(im)) >= cfg["min_category_images"]:
                dx = a["dx"][ix].astype(float)
                ga = ma["contrast"]["groups"][0]
                gb = mb["contrast"]["groups"][0]
                qa = a["dz"][ix, ga] @ a["atoms"][ga]
                qb = b["dz"][ix, gb] @ b["atoms"][gb]
                item.update(
                    den=np.bincount(im, weights=(dx**2).sum(1), minlength=n),
                    ea=np.bincount(im, weights=((dx - qa) ** 2).sum(1), minlength=n),
                    eb=np.bincount(im, weights=((dx - qb) ** 2).sum(1), minlength=n),
                )
        prepared.append(item)
    rng = np.random.default_rng(cfg["fresh_test_seed"] + 90)
    ap = []
    cap = []
    for _ in range(cfg["bootstrap"]):
        count = np.bincount(rng.integers(n, size=n), minlength=n)
        w = count[a["image_index"]]
        aa = []
        cc = []
        for p in prepared:
            c = p["category_index"]
            if "det_a" in p:
                x = metrics.ap(p["det_a"], ya[:, c], va[:, c], w)
                y = metrics.ap(p["det_b"], ya[:, c], va[:, c], w)
                if x is not None and y is not None:
                    aa.append(x - y)
            if "den" in p:
                den = float(count @ p["den"])
                if den > 1e-12:
                    cc.append(float(count @ (p["eb"] - p["ea"]) / den))
        if aa:
            ap.append(np.mean(aa))
        if cc:
            cap.append(np.mean(cc))
    return dict(
        macro_single_group_ap_gain_ci95=metrics.interval(ap),
        macro_single_group_contrast_gain_ci95=metrics.interval(cap),
        detection_categories=sum("det_a" in x for x in prepared),
        contrast_categories=sum("den" in x for x in prepared),
        note="Exploratory paired image bootstrap; excludes seed uncertainty and multiple-comparison adjustment.",
    )


def plot(store, reports):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    methods = list(dict.fromkeys(r["summary"]["method"] for r in reports))
    fig, axes = plt.subplots(
        len(methods), 3, figsize=(18, 4.3 * len(methods)), squeeze=False
    )
    labels = {
        "native": "Native",
        "continued_reconstruction": "Continued BSF",
        "supervised_fixed": "Supervised fixed",
        "supervised_adaptive": "Adaptive",
        "supervised_fixed_equal_search_compute": "Fixed: search compute",
    }
    labels.update(
        anchored_supervised_fixed="Anchored fixed",
        anchored_supervised_adaptive="Anchored adaptive",
        anchored_supervised_fixed_equal_search_compute="Anchored fixed: search compute",
    )
    for i, method in enumerate(methods):
        rows = [r["summary"] for r in reports if r["summary"]["method"] == method]
        x = np.arange(len(rows))
        names = [labels[r["arm"]] for r in rows]
        for ax, field, title in zip(
            axes[i],
            (
                "mean_single_group_ap",
                "mean_single_group_contrast_capture",
                "reconstruction",
            ),
            ("One-group object AP", "One-group object contrast", "Reconstruction VE"),
        ):
            values = [
                r["reconstruction"]["ve"] if field == "reconstruction" else r[field]
                for r in rows
            ]
            ax.bar(
                x,
                [np.nan if v is None else v for v in values],
                color=[
                    "#92a7ba",
                    "#bac7c5",
                    "#86b8af",
                    "#008e89",
                    "#b8a2cf",
                    "#e4bb72",
                    "#db7b32",
                    "#a082bd",
                ][: len(rows)],
            )
            ax.set_xticks(x, names, rotation=35, ha="right", fontsize=8)
            ax.set_title(method + ": " + title)
            ax.axhline(0, color="#444", lw=0.6)
            ax.grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Fresh COCO images: supervised adaptive versus matched fixed-layout controls",
        fontsize=13,
    )
    fig.tight_layout()
    folder = store.results / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / "adaptation_comparison.png", dpi=180)
    fig.savefig(folder / "adaptation_comparison.pdf")
    plt.close(fig)


def preflight(source, choices, packs, cfg):
    base = source.config["config"]
    measure = []
    for choice in choices:
        state = source.load(choice["key"])
        w = state["best"] if choice["sparsity_matched"] else state["fallback"]
        m = core.new_model(choice["method"], base, w, cfg["device"])
        opt = torch.optim.Adam(m.parameters(), lr=cfg["lr"])
        x = torch.as_tensor(
            packs["train"]["x"][: cfg["images_per_update"]].reshape(-1, m.d),
            device=cfg["device"],
        )
        times = []
        for i in range(6):
            started = time.monotonic()
            m.train()
            loss, _ = m.loss(x, target=x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            m.normalize_decoder()
            if cfg["device"].startswith("cuda"):
                torch.cuda.synchronize()
            if i > 1:
                times.append(time.monotonic() - started)
        per_variant = cfg["warm_updates"] + cfg["rounds"] * cfg["refit_updates"] * (
            1
            + cfg["candidates_per_round"]
            + 1
            + (
                cfg["candidates_per_round"] + 1
                if cfg["equal_search_compute_control"]
                else 0
            )
        )
        variant_count = len(
            cfg.get(
                "adaptation_variants",
                ["unanchored", "anchored"] if cfg["anchor_ablation"] else ["anchored"],
            )
        )
        u = (
            cfg["warm_updates"]
            + cfg["rounds"] * cfg["refit_updates"]
            + per_variant * variant_count
        )
        measure.append(
            dict(
                method=choice["method"],
                native_seconds_per_update=float(np.median(times)),
                max_gradient_updates=u,
                optimistic_native_gradient_minutes=u * float(np.median(times)) / 60,
            )
        )
        del m, opt
    return dict(
        measurements=measure,
        max_final_models=len(choices) * (2 + 3 * variant_count),
        excludes="Supervised loss, anchor replay/Gram matrices, component-gradient diagnostics, validation, proposal audits, Drive saves, new-image DINO extraction and final audits. Native timings are a lower-work reference, not an ETA.",
    )


def export(store, source):
    path = (
        store.output / f"agf_coco_supervised_adaptation_{store.run_id[:12]}_results.zip"
    )
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in store.results.rglob("*"):
            if p.is_file():
                z.write(p, "results/" + p.relative_to(store.results).as_posix())
        for p in Path(source).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                z.write(p, "source/" + p.relative_to(source).as_posix())
    return path
