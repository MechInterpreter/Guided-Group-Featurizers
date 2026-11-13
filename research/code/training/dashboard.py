"""One metric implementation for every method; label/selection locks are explicit."""

import copy, hashlib, json
from pathlib import Path
import numpy as np
import original_concentration as original
from category_ap import select_by_calibration_ap, nonnegative_ap


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def pooled_native(maps, groups=75):
    x = np.asarray(maps)
    if x.ndim != 3 or x.shape[1:] != (196, groups) or not np.isfinite(x).all():
        raise ValueError("Exact declared native response shape required")
    if groups == 75:
        return original.mean_top4(x)
    # Native category-split SLiCS reference has37/38; do not pad/slice its bank.
    if groups not in [37, 38]:
        raise ValueError("Unsupported native reference capacity")
    return np.partition(x, -4, axis=1)[:, -4:, :].mean(1)


def validate_splits(calibration_ids, audit_ids):
    a, b = list(calibration_ids), list(audit_ids)
    if (
        not a
        or not b
        or len(set(a)) != len(a)
        or len(set(b)) != len(b)
        or set(a) & set(b)
    ):
        raise ValueError("Duplicate, empty or overlapping split IDs")


def calibration_selection(
    responses, ids, calibration_ids, labels, source_lock, groups=75
):
    if (
        source_lock.get("presence_complete") is not True
        or not source_lock.get("presence_sha256")
        or not source_lock.get("protocol_sha256")
    ):
        raise ValueError("Complete independent presence and protocol lock required")
    x = np.asarray(responses)
    ids = list(ids)
    if (
        x.shape != (len(ids), groups)
        or len(set(ids)) != len(ids)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid native pooled bank")
    if len(set(calibration_ids)) != len(calibration_ids) or not set(
        calibration_ids
    ) <= set(ids):
        raise ValueError("Calibration identity differs")
    lookup = {v: i for i, v in enumerate(ids)}
    support = original.support(labels, calibration_ids)
    selection = dict(
        description_only=True,
        candidate_units=list(range(groups)),
        description_support=support,
        quantiles=[],
    )
    for q in original.QUANTILES:
        row = dict(
            negative_quantile=q,
            status="complete"
            if support["eligible"]
            else "insufficient_description_support",
        )
        if support["eligible"]:
            pos = [lookup[i] for i in calibration_ids if labels.get(i) == "yes"]
            neg = [lookup[i] for i in calibration_ids if labels.get(i) == "no"]
            threshold = np.quantile(x[neg], q, axis=0, method="higher")
            pred = x > threshold
            quality = 0.5 * (pred[pos].mean(0) + 1 - pred[neg].mean(0))
            rank = sorted(range(groups), key=lambda u: (-float(quality[u]), u))
            row.update(
                thresholds=threshold.tolist(),
                balanced_accuracy_by_unit=quality.tolist(),
                ranked_units=rank,
                selections=[
                    dict(
                        budget=b,
                        groups=rank[: min(b, groups)],
                        groups_used=min(b, groups),
                    )
                    for b in original.BUDGETS
                ],
            )
        else:
            row.update(
                thresholds=None,
                balanced_accuracy_by_unit=None,
                ranked_units=[],
                selections=[
                    dict(budget=b, groups=[], groups_used=0) for b in original.BUDGETS
                ],
            )
        selection["quantiles"].append(row)
    selection.update(
        calibration_ids=list(calibration_ids),
        groups=groups,
        source_lock=copy.deepcopy(source_lock),
    )
    selection["selection_sha256"] = digest(selection)
    return selection


def audit_concentration(selection, responses, ids, audit_ids, labels, selection_lock):
    check = copy.deepcopy(selection)
    expected = check.pop("selection_sha256")
    if (
        digest(check) != expected
        or selection_lock.get("selection_sha256") != expected
        or not selection_lock.get("locked_before_audit")
    ):
        raise ValueError("Calibration selection lock is absent or changed")
    validate_splits(selection["calibration_ids"], audit_ids)
    x = np.asarray(responses)
    ids = list(ids)
    g = selection["groups"]
    if (
        x.shape != (len(ids), g)
        or len(set(ids)) != len(ids)
        or not np.isfinite(x).all()
        or not set(audit_ids) <= set(ids)
    ):
        raise ValueError("Audit bank identity differs")
    support = original.support(labels, audit_ids)
    known = [i for i in audit_ids if labels.get(i) in ["yes", "no"]]
    positive = np.array([labels[i] == "yes" for i in known])
    lookup = {v: i for i, v in enumerate(ids)}
    rows = []
    for q in selection["quantiles"]:
        for chosen in q["selections"]:
            status = (
                q["status"]
                if q["status"] != "complete"
                else (
                    "complete" if support["eligible"] else "insufficient_audit_support"
                )
            )
            row = dict(
                status=status,
                negative_quantile=q["negative_quantile"],
                budget=chosen["budget"],
                groups=chosen["groups"],
                candidate_groups=g,
                description_support=selection["description_support"],
                audit_support=support,
                metrics=None,
            )
            if status == "complete":
                units = chosen["groups"]
                pred = (
                    x[[lookup[i] for i in known]][:, units]
                    > np.asarray(q["thresholds"])[units]
                )
                row["metrics"] = original.or_metrics(pred, positive, units)
            rows.append(row)
    return rows


def coherence_counts(records, planned_groups=75):
    if len(records) != planned_groups or sorted(r["unit"] for r in records) != list(
        range(planned_groups)
    ):
        raise ValueError("Complete native-unit denominator required")
    result = {}
    for endpoint in ["semantic", "visual"]:
        counts = {k: 0 for k in ["yes", "no", "unclear", "missing"]}
        for r in records:
            if len(r["pairs"]) != 3:
                raise ValueError("Exactly three frozen pairs per group")
            for p in r["pairs"]:
                code = p.get(endpoint, "missing")
                if code not in counts:
                    raise ValueError("Unknown coherence code")
                counts[code] += 1
        n = planned_groups * 3
        result[endpoint] = dict(
            counts=counts,
            denominator=n,
            confirmed=counts["yes"] / n,
            unresolved_upper=(counts["yes"] + counts["unclear"] + counts["missing"])
            / n,
        )
    return result


def alignment_counts(codes, planned=6):
    allowed = ["target", "related_context", "other", "mixed", "unclear", "missing"]
    if len(codes) > planned or any(c not in allowed for c in codes):
        raise ValueError("Invalid alignment judgments")
    counts = {k: codes.count(k) for k in allowed}
    counts["missing"] += planned - len(codes)
    return dict(
        counts=counts,
        planned=planned,
        confirmed=counts["target"] / planned,
        unresolved_upper=(
            counts["target"] + counts["mixed"] + counts["unclear"] + counts["missing"]
        )
        / planned,
    )
