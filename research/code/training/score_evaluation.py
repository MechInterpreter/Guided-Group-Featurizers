"""Shared numerical and annotation scorers. No model fitting or winner selection."""

import argparse, copy, json
from pathlib import Path
import numpy as np
from eval_common import *
import dashboard
from category_ap import nonnegative_ap


def mask_labels(root, role):
    meta = read(Path(root) / "labels" / f"{role}_masks.json")
    p = resolve(root, meta["path"])
    if sha(p) != meta["sha256"]:
        raise ValueError("Annotation masks changed")
    with np.load(p) as z:
        f = z["fraction"].astype(np.float32) / 255.0
        crowd = z["crowd"]
        ids = z["ids"].tolist()
    y = f >= 0.5
    valid = ((f >= 0.5) | (f <= 0.1)) & (crowd[:, :, None] <= 0.1)
    return ids, y.reshape(-1, y.shape[-1]), valid.reshape(-1, y.shape[-1])


def choose_mask_AP(scores, y, valid):
    G = scores.shape[-1]
    flat = scores.reshape(-1, G)
    C = y.shape[-1]
    table = np.full((G, C), np.nan)
    supported = (y & valid).sum(0) > 0
    supported &= ((~y) & valid).sum(0) > 0
    for g in range(G):
        table[g] = nonnegative_ap(flat[:, g], y, valid)
    selected = [
        int(np.nanargmax(table[:, c])) if supported[c] else None for c in range(C)
    ]
    return selected, table


def lock_numeric_readouts(root, bank, categories):
    """Called after CALIBRATION chunks, before opening that bank's audit loop."""
    folder = Path(root) / "responses" / bank["bank_id"]
    path = folder / "NUMERIC_READOUT_LOCK.json"
    cal = read_maps(root, bank, "calibration")
    ids, y, v = mask_labels(root, "calibration")
    if ids != cal["ids"]:
        raise ValueError("Calibration mask/response identity differs")
    units, table = choose_mask_AP(cal["scores"], y, v)
    capture = []
    for j, c in enumerate(categories):
        good = cal["capture_count"][j] >= 3 and cal["capture_den"][j] > 1e-12
        capture.append(int(cal["capture_errors"][j].argmin()) if good else None)
    result = dict(
        schema="all75-calibration-numeric-readouts-v1",
        bank_id=bank["bank_id"],
        weights_sha=bank["weights_sha"],
        calibration_only=True,
        categories=categories,
        calibration_ids=ids,
        AP_units=units,
        Capture_units=capture,
        calibration_AP=table,
        calibration_capture_count=cal["capture_count"],
        calibration_capture_den=cal["capture_den"],
        source_chunk_sha256=[r["sha256"] for r in bank["splits"]["calibration"]],
        AP_selector="Maximum category-mask patch AP; tied zero retained; smallest unit ID breaks ties",
        Capture_selector="Minimum regional reconstruction error on calibration only; separate control, never target alignment selector",
    )
    save(path, result)
    return path


def category_rows(bank, lock, audit, y, valid, categories):
    flat = audit["scores"].reshape(-1, bank["groups"])
    rows = []
    n = len(audit["ids"])
    for j, c in enumerate(categories):
        unit = lock["AP_units"][j]
        positive = y[:, j] & valid[:, j]
        negative = ~y[:, j] & valid[:, j]
        npos = int(positive.sum())
        nneg = int(negative.sum())
        result = dict(
            category_index=c,
            AP_unit=unit,
            positive_patches=npos,
            negative_patches=nneg,
            patch_AP=None,
            prevalence=npos / (npos + nneg) if npos + nneg else None,
            AP_minus_prevalence=None,
            normalized_AP_lift=None,
            within_target_images_AP=None,
            status="unsupported_calibration"
            if unit is None
            else ("complete" if npos and nneg else "unsupported_audit"),
        )
        if result["status"] == "complete":
            s = flat[:, unit]
            ap = nonnegative_ap(s, y[:, j], valid[:, j])
            prev = result["prevalence"]
            images = positive.reshape(n, 196).any(1)
            within = valid[:, j] & np.repeat(images, 196)
            result.update(
                patch_AP=ap,
                AP_minus_prevalence=ap - prev,
                normalized_AP_lift=(ap - prev) / (1 - prev),
                within_target_images_AP=nonnegative_ap(s, y[:, j], within),
            )
        cu = lock["Capture_units"][j]
        den = float(audit["capture_den"][j])
        count = int(audit["capture_count"][j])
        result.update(
            Capture_unit=cu,
            Capture_objects=count,
            regional_Capture=1 - float(audit["capture_errors"][j, cu]) / den
            if cu is not None and den > 1e-12 and count >= 3
            else None,
            AP_uncertainty="Point estimate conditional on frozen calibration selector; no model-population confidence claim",
        )
        rows.append(clean(result))
    return rows


def numeric(root):
    root = Path(root)
    collection = read(root / "COLLECTION_COMPLETE.json")
    inputs = read(root / "INPUTS.json")
    if (
        not collection["complete"]
        or sha(root / "INPUTS.json") != collection["inputs_sha256"]
    ):
        raise ValueError("Input/collection lock differs")
    ids, y, v = mask_labels(root, "audit")
    results = []
    for bank in collection["banks"]:
        lock_path = root / "responses" / bank["bank_id"] / "NUMERIC_READOUT_LOCK.json"
        if sha(lock_path) != bank["numeric_readout_sha256"]:
            raise ValueError("Calibration readouts changed")
        lock = read(lock_path)
        audit = read_maps(root, bank, "audit")
        if (
            audit["ids"] != ids
            or lock["calibration_ids"] != inputs["ids"]["calibration"]
        ):
            raise ValueError("Scoring split IDs differ")
        rows = category_rows(bank, lock, audit, y, v, inputs["categories"])
        controls = audit["controls"]
        maps = audit["scores"]
        additive = bank["kind"] != "slics"
        sse = float(controls[:, 0].sum()) if additive else None
        resource = dict(
            core_parameters=bank["core_parameters"],
            head_parameters=2 * len(bank.get("guide", inputs["categories"]))
            if bank["kind"] in ["ggf", "supervised_vanilla"]
            else 0,
            collection_wall_seconds=bank["collection_wall_seconds"],
            observed_peak_GPU_bytes=bank["observed_peak_GPU_bytes"],
            active_groups_per_patch=float(controls[:, 3].sum() / (len(ids) * 196)),
            dead_groups=int((~(maps > 1e-6).any((0, 1))).sum()),
            active_patch_frequency=(maps > 1e-6).mean((0, 1)),
            reconstruction_MSE=sse / collection["audit_values"] if additive else None,
            reconstruction_VE=1 - sse / collection["audit_total_variation"]
            if additive and collection["audit_total_variation"] > 0
            else None,
            reconstruction_status="native_additive_BSF"
            if additive
            else "not_defined_for_independent_SLICS_concept_projections; not summed",
            training_seconds=bank.get("training_seconds"),
            training_resource_scope="Training receipt if newly fitted; reused source timing may be unavailable, not zero",
        )
        result = dict(
            bank_id=bank["bank_id"],
            kind=bank["kind"],
            regime=bank["regime"],
            seed=bank["seed"],
            groups=bank["groups"],
            evaluation_folds=bank["evaluation_folds"],
            capacity_matched=bank["capacity_matched"],
            categories=rows,
            controls=resource,
            semantic_coherence_status="pending_separate_blinded_labels",
            target_alignment_status="pending_presence_based_selector_and_visual_labels",
        )
        save(root / "numeric" / f"{bank['bank_id']}.json", result)
        results.append(result)
        print("NUMERIC SCORE COMPLETE", bank["bank_id"], flush=True)
    save(
        root / "NUMERIC_COMPLETE.json",
        dict(
            complete=True,
            collection_sha256=sha(root / "COLLECTION_COMPLETE.json"),
            results=results,
            scope="Numerical annotated alignment and controls only; no imputed semantic or concentration results",
        ),
    )


def read_presence(root, role):
    p = Path(root) / "presence" / f"{role}_LOCK.json"
    lock = read(p)
    qa = read(Path(root) / "PRESENCE_QA.json")
    if not qa.get("complete") or qa["canonical_lock_sha256"][role] != sha(p):
        raise ValueError("Independent fixed presence-QA lock required")
    if sha(Path(root) / "reviews/presence_qa/labels.json") != qa["labels_sha256"]:
        raise ValueError("Presence QA source changed")
    if (
        not lock.get("complete")
        or lock["role"] != role
        or lock["definitions_sha256"] != sha(Path(root) / "TARGET_DEFINITIONS.json")
    ):
        raise ValueError("Complete independent presence lock required")
    for f in lock["sources"]:
        if sha(resolve(root, f["path"])) != f["sha256"]:
            raise ValueError("Presence source changed")
    return lock


def select_presence(root):
    root = Path(root)
    collection = read(root / "COLLECTION_COMPLETE.json")
    inputs = read(root / "INPUTS.json")
    cal = read_presence(root, "calibration")
    if cal["ids"] != inputs["ids"]["calibration"]:
        raise ValueError("Presence image order differs")
    locks = []
    for bank in collection["banks"]:
        a = read_pooled(root, bank, "calibration")
        rows = []
        for c in inputs["categories"]:
            labels = {int(i): s for i, s in cal["labels"][str(c)].items()}
            sel = dashboard.calibration_selection(
                a["pooled"],
                a["ids"],
                a["ids"],
                labels,
                dict(
                    presence_complete=True,
                    presence_sha256=sha(root / "presence/calibration_LOCK.json"),
                    protocol_sha256=sha(root / "PROTOCOL.json"),
                ),
                bank["groups"],
            )
            rows.append(dict(category_index=c, selection=sel))
        p = root / "concentration" / bank["bank_id"] / "SELECTION_LOCK.json"
        save(
            p,
            dict(
                bank_id=bank["bank_id"],
                calibration_only=True,
                calibration_presence_sha256=sha(
                    root / "presence/calibration_LOCK.json"
                ),
                rows=rows,
            ),
        )
        locks.append(
            dict(bank_id=bank["bank_id"], path=str(p.relative_to(root)), sha256=sha(p))
        )
    save(
        root / "CONCENTRATION_SELECTION_COMPLETE.json",
        dict(
            complete=True,
            locks=locks,
            source="Independent calibration presence only; no audit scores consulted",
        ),
    )


def concentration(root):
    root = Path(root)
    collection = read(root / "COLLECTION_COMPLETE.json")
    inputs = read(root / "INPUTS.json")
    cal = read_presence(root, "calibration")
    audit = read_presence(root, "audit")
    if audit["ids"] != inputs["ids"]["audit"]:
        raise ValueError("Audit presence order differs")
    locks = read(root / "CONCENTRATION_SELECTION_COMPLETE.json")
    by = {r["bank_id"]: r for r in locks["locks"]}
    results = []
    for bank in collection["banks"]:
        ref = by[bank["bank_id"]]
        path = resolve(root, ref["path"])
        if sha(path) != ref["sha256"]:
            raise ValueError("Concentration selector changed")
        lock = read(path)
        if lock["calibration_presence_sha256"] != sha(
            root / "presence/calibration_LOCK.json"
        ):
            raise ValueError("Calibration presence changed after selection")
        a = read_pooled(root, bank, "audit")
        rows = []
        for row in lock["rows"]:
            c = row["category_index"]
            s = row["selection"]
            labels = {int(i): v for i, v in audit["labels"][str(c)].items()}
            # The pure kernel checks disjointness against the true calibration IDs.
            curves = dashboard.audit_concentration(
                s,
                a["pooled"],
                a["ids"],
                a["ids"],
                labels,
                dict(selection_sha256=s["selection_sha256"], locked_before_audit=True),
            )
            rows.append(dict(category_index=c, curves=curves))
        result = dict(
            bank_id=bank["bank_id"],
            rows=rows,
            selection_lock_sha256=ref["sha256"],
            audit_presence_sha256=sha(root / "presence/audit_LOCK.json"),
        )
        save(root / "concentration" / bank["bank_id"] / "RESULTS.json", result)
        results.append(result)
    save(
        root / "CONCENTRATION_COMPLETE.json",
        dict(
            complete=True,
            results=results,
            scope="Category-conditioned response concentration, not decoder information recovery or semantic coherence",
        ),
    )


def majority(values):
    if any(v not in ["yes", "no", "unclear"] for v in values):
        raise ValueError("Unknown pair label")
    return (
        "yes"
        if values.count("yes") >= 2
        else ("no" if values.count("no") >= 2 else "unclear")
    )


def coherence(root):
    from prepare_evaluation_reviews import verify_pack

    root = Path(root)
    verify_pack(root / "reviews/coherence")
    public = read(root / "reviews/coherence/PUBLIC.json")
    private = read(root / "reviews/coherence/PRIVATE.json")
    contexts = []
    sources = []
    expected = {
        b["bank_id"]: b["groups"]
        for b in read(root / "COLLECTION_COMPLETE.json")["banks"]
    }
    for bank, groups in expected.items():
        if sorted(r["unit"] for r in private["cases"] if r["bank_id"] == bank) != list(
            range(groups)
        ):
            raise ValueError("Complete native bank required")
    for name in ["context1", "context2", "context3"]:
        p = root / "reviews/coherence/labels" / f"{name}.json"
        if not p.exists():
            raise FileNotFoundError("Fresh blinded context pending: " + str(p))
        d = read(p)
        if (
            d["public_sha256"] != sha(root / "reviews/coherence/PUBLIC.json")
            or not d.get("method_blind")
            or not d.get("reviewer")
        ):
            raise ValueError("Reviewer provenance/blinding absent")
        cases = {r["case_id"]: r for r in d["cases"]}
        for source in d.get("sources", []):
            if sha(resolve(root, source["path"])) != source["sha256"]:
                raise ValueError("Raw blinded reviewer batch changed")
        if len(cases) != len(d["cases"]) or set(cases) != {
            r["case_id"] for r in public["cases"]
        }:
            raise ValueError("Full fixed group denominator required")
        contexts.append(cases)
        sources.append(
            dict(path=str(p.relative_to(root)), sha256=sha(p), reviewer=d["reviewer"])
        )
    if len({s["reviewer"] for s in sources}) != 3:
        raise ValueError("Record three distinct fresh reviewer contexts")
    bybank = {}
    for case in private["cases"]:
        pairs = []
        for pair in range(3):
            values = {endpoint: [] for endpoint in ["semantic", "visual"]}
            for context in contexts:
                p = context[case["case_id"]]["pairs"]
                if sorted(r["pair"] for r in p) != [0, 1, 2]:
                    raise ValueError("All three fixed pairs required")
                p = next(r for r in p if r["pair"] == pair)
                for endpoint in values:
                    code = p[endpoint]
                    if case["invalid_pairs"][pair] and code != "unclear":
                        raise ValueError(
                            "Missing/zero-response pair must remain unclear"
                        )
                    if code == "yes" and not p.get("shared_meaning", "").strip():
                        raise ValueError(
                            "A positive interpretation needs its stated shared meaning"
                        )
                    values[endpoint].append(code)
            pairs.append(dict(pair=pair, **{e: majority(v) for e, v in values.items()}))
        bybank.setdefault(case["bank_id"], []).append(
            dict(unit=case["unit"], pairs=pairs)
        )
    counts = {
        b: dashboard.coherence_counts(rows, len(rows)) for b, rows in bybank.items()
    }
    save(
        root / "COHERENCE_COMPLETE.json",
        dict(
            complete=True,
            bank_results=counts,
            sources=sources,
            public_sha256=sha(root / "reviews/coherence/PUBLIC.json"),
            scope="AI majority specific-semantic and visual pair endpoints; contexts are not independent human raters",
        ),
    )


def alignment(root, role):
    from prepare_evaluation_reviews import verify_pack

    root = Path(root)
    folder = root / "reviews" / ("alignment_" + role)
    verify_pack(folder)
    public = read(folder / "PUBLIC.json")
    private = read(folder / "PRIVATE.json")
    p = folder / "labels.json"
    d = read(p)
    if (
        d["public_sha256"] != sha(folder / "PUBLIC.json")
        or not d.get("method_blind")
        or not d.get("reviewer")
    ):
        raise ValueError("Blinded alignment provenance required")
    cases = {r["case_id"]: r for r in d["cases"]}
    if len(cases) != len(d["cases"]) or set(cases) != {
        r["case_id"] for r in public["cases"]
    }:
        raise ValueError("Complete planned alignment views required")
    rows = []
    for r in private["cases"]:
        codes = cases[r["case_id"]]["codes"]
        if len(codes) != 6:
            raise ValueError("Six planned alignment slots required")
        for j, bad in enumerate(r["invalid_views"]):
            if bad and codes[j] not in ["unclear", "missing"]:
                raise ValueError("Unavailable evidence cannot be a confirmed target")
        rows.append(
            dict(
                bank_id=r["bank_id"],
                category_index=r["category_index"],
                unit=r["unit"],
                result=dashboard.alignment_counts(codes),
            )
        )
    save(
        root / f"ALIGNMENT_{role.upper()}_LOCK.json",
        dict(
            complete=True,
            role=role,
            rows=rows,
            labels_sha256=sha(p),
            public_sha256=sha(folder / "PUBLIC.json"),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument(
        "--phase",
        choices=[
            "numeric",
            "select-presence",
            "concentration",
            "coherence",
            "alignment-calibration",
            "alignment-audit",
        ],
        required=True,
    )
    a = p.parse_args()
    if a.phase.startswith("alignment-"):
        alignment(a.root, a.phase.split("-")[1])
    else:
        {
            "numeric": numeric,
            "select-presence": select_presence,
            "concentration": concentration,
            "coherence": coherence,
        }[a.phase](a.root)
