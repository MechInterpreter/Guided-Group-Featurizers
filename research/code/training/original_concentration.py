"""Pure saved-response concentration calculations. No inference or production I/O on import."""

import hashlib, itertools, json
from pathlib import Path
import numpy as np

QUANTILES = (0.95, 0.90)
BUDGETS = (1, 2, 4, 8)
UNITS = tuple(range(75))
STATES = {"yes", "no", "unclear", "missing", "unreviewed"}
ALIGNMENT_STATES = {"target", "related_context", "other", "mixed", "unclear"}


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_verified(path):
    path = Path(path)
    side = next(
        (
            p
            for p in [path.with_suffix(".sha.json"), Path(str(path) + ".sha.json")]
            if p.exists()
        ),
        None,
    )
    if side is None:
        raise ValueError("Missing SHA sidecar: " + str(path))
    stamp = json.loads(side.read_text())
    if file_sha(path) != stamp.get("sha256", stamp.get("sha")):
        raise ValueError("SHA mismatch: " + str(path))
    return json.loads(path.read_text())


def mean_top4(scores):
    """Mean of largest four SIGNED responses among all196 positions."""
    x = np.asarray(scores)
    if x.ndim != 3 or x.shape[1] != 196 or x.shape[2] < 75 or not np.isfinite(x).all():
        raise ValueError("Expected finite images x196 x at least75 native groups")
    return np.partition(x[:, :, :75], -4, axis=1)[:, -4:, :].mean(axis=1)


def validate_responses(responses, ids):
    x = np.asarray(responses, dtype=float)
    ids = list(map(int, ids))
    if (
        x.shape != (len(ids), 75)
        or len(set(ids)) != len(ids)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected unique image IDs and finite images x75 responses")
    return x, ids, {iid: i for i, iid in enumerate(ids)}


def expand_presence(categories, manifest, records, require_complete=False):
    """Independent crop labels only: never derive semantic presence from responses."""
    cats = (
        categories.get("categories", categories)
        if isinstance(categories, dict)
        else categories
    )
    images = (
        manifest.get("images", manifest) if isinstance(manifest, dict) else manifest
    )
    if len(cats) != 75:
        raise ValueError("Exactly75 frozen categories required")
    keys = [c["category_key"] for c in cats]
    index_to_key = {c["category_index"]: c["category_key"] for c in cats}
    if len(set(keys)) != 75 or len(index_to_key) != 75:
        raise ValueError("Category IDs must be unique")
    by_id = {int(q["image_id"]): q for q in images}
    if len(by_id) != len(images):
        raise ValueError("Duplicate manifest image")
    matrix = {k: {iid: "unreviewed" for iid in by_id} for k in keys}
    seen = set()
    payload = records.get("records", records) if isinstance(records, dict) else records
    for row in payload:
        iid = int(row["image_id"])
        if iid not in by_id or iid in seen:
            raise ValueError("Unknown/duplicate presence image")
        seen.add(iid)
        available = bool(by_id[iid]["available"])
        status = row.get("status", row.get("review_status"))
        if status == "missing":
            if available:
                raise ValueError("Available crop cannot be labeled missing")
            if any(row.get(k) for k in ["yes", "no", "unclear"]):
                raise ValueError("Missing crop cannot have semantic judgments")
            if "missing" in row and set(row["missing"]) != set(index_to_key):
                raise ValueError("Missing crop must preserve all75 missing indices")
            for k in keys:
                matrix[k][iid] = "missing"
            continue
        if not available or status != "reviewed":
            raise ValueError("Require actual available reviewed crop")
        if not (row.get("all_definitions_considered") or row.get("all75_considered")):
            raise ValueError(
                "No omitted-category negatives without all75 consideration"
            )

        def mapped(values):
            result = []
            for value in values:
                key = index_to_key.get(value) if isinstance(value, int) else value
                if key not in keys:
                    raise ValueError("Unknown category identifier")
                result.append(key)
            if len(set(result)) != len(result):
                raise ValueError("Duplicate category judgment")
            return set(result)

        yes = mapped(row.get("yes", []))
        unclear = mapped(row.get("unclear", []))
        if yes & unclear:
            raise ValueError("Yes and unclear overlap")
        if row.get("missing"):
            raise ValueError("Available crop cannot have missing category judgments")
        if "no" in row and mapped(row["no"]) != set(keys) - yes - unclear:
            raise ValueError("Explicit no list is not the remaining75 categories")
        for k in keys:
            matrix[k][iid] = (
                "yes" if k in yes else ("unclear" if k in unclear else "no")
            )
    if require_complete and seen != set(by_id):
        raise ValueError("All planned400 images must be reviewed or missing")
    return matrix


def support(labels, half_ids):
    counts = {s: 0 for s in ["yes", "no", "unclear", "missing", "unreviewed"]}
    for iid in half_ids:
        state = labels.get(int(iid), "unreviewed")
        if state not in STATES:
            raise ValueError("Invalid presence state")
        counts[state] += 1
    return dict(
        planned=len(half_ids),
        **counts,
        eligible=counts["yes"] >= 5 and counts["no"] >= 10,
    )


def calibrate_description(responses, ids, description_ids, labels):
    """Fit thresholds/ranking on description labels ONLY; all75 are candidates."""
    x, ids, lookup = validate_responses(responses, ids)
    description_ids = list(map(int, description_ids))
    if len(set(description_ids)) != len(description_ids) or not set(
        description_ids
    ) <= set(ids):
        raise ValueError("Invalid description IDs")
    counts = support(labels, description_ids)
    result = dict(
        description_only=True,
        candidate_units=list(UNITS),
        description_support=counts,
        quantiles=[],
    )
    for q in QUANTILES:
        row = dict(
            negative_quantile=q,
            status="complete"
            if counts["eligible"]
            else "insufficient_description_support",
        )
        if counts["eligible"]:
            pos = [lookup[i] for i in description_ids if labels.get(i) == "yes"]
            neg = [lookup[i] for i in description_ids if labels.get(i) == "no"]
            threshold = np.quantile(x[neg], q, axis=0, method="higher")
            detections = x > threshold
            quality = 0.5 * (
                detections[pos].mean(axis=0) + 1 - detections[neg].mean(axis=0)
            )
            order = sorted(UNITS, key=lambda u: (-float(quality[u]), u))
            row.update(
                thresholds=threshold.tolist(),
                balanced_accuracy_by_unit=quality.tolist(),
                ranked_units=order,
                selections=[
                    dict(budget=b, groups=order[: min(b, 75)], groups_used=min(b, 75))
                    for b in BUDGETS
                ],
            )
        else:
            row.update(
                thresholds=None,
                balanced_accuracy_by_unit=None,
                ranked_units=[],
                selections=[dict(budget=b, groups=[], groups_used=0) for b in BUDGETS],
            )
        result["quantiles"].append(row)
    return result


def _overlap(pred, positive):
    values = []
    zero = 0
    for a, b in itertools.combinations(range(pred.shape[1]), 2):
        aa = pred[positive, a]
        bb = pred[positive, b]
        union = int(np.sum(aa | bb))
        if union:
            values.append(int(np.sum(aa & bb)) / union)
        else:
            zero += 1
    return dict(
        mean_jaccard=float(np.mean(values)) if values else None,
        informative_pairs=len(values),
        zero_union_pairs=zero,
        planned_pairs=pred.shape[1] * (pred.shape[1] - 1) // 2,
    )


def or_metrics(detections, positive, groups):
    """OR only. Exact incremental and leave-one-group unique coverage counts."""
    pred = np.asarray(detections, dtype=bool)
    positive = np.asarray(positive, dtype=bool)
    if (
        pred.shape != (len(positive), len(groups))
        or positive.sum() < 5
        or (~positive).sum() < 10
    ):
        raise ValueError(
            "Audit metric requires at least5 positive/10 negative and matching groups"
        )
    npos = int(positive.sum())
    nneg = int((~positive).sum())
    union = pred.any(axis=1)
    tp = int(np.sum(union & positive))
    fp = int(np.sum(union & ~positive))
    prior = np.zeros(len(positive), dtype=bool)
    increments = []
    unique = []
    for j, u in enumerate(groups):
        added = pred[:, j] & ~prior
        prior |= pred[:, j]
        increments.append(
            dict(
                unit=int(u),
                rank=j + 1,
                new_positive=int(np.sum(added & positive)),
                new_negative=int(np.sum(added & ~positive)),
                positive_increment=int(np.sum(added & positive)) / npos,
                negative_increment=int(np.sum(added & ~positive)) / nneg,
                cumulative_positive=int(np.sum(prior & positive)),
                cumulative_negative=int(np.sum(prior & ~positive)),
            )
        )
        other = np.delete(pred, j, axis=1).any(axis=1)
        exclusive = pred[:, j] & ~other
        unique.append(
            dict(
                unit=int(u),
                positive_count=int(np.sum(exclusive & positive)),
                negative_count=int(np.sum(exclusive & ~positive)),
                positive_rate=int(np.sum(exclusive & positive)) / npos,
                negative_rate=int(np.sum(exclusive & ~positive)) / nneg,
            )
        )
    return dict(
        positive_count=npos,
        negative_count=nneg,
        true_positive=tp,
        false_positive=fp,
        OR_coverage=tp / npos,
        OR_FPR=fp / nneg,
        positive_prediction_precision=tp / (tp + fp) if tp + fp else None,
        positive_overlap=_overlap(pred, positive),
        negative_overlap=_overlap(pred, ~positive),
        incremental_coverage=increments,
        unique_coverage=unique,
    )


def evaluate_category_synthetic_or_after_gates(
    selection, responses, ids, audit_ids, labels
):
    """Pure evaluation kernel; production callers MUST first verify_artifact_gates."""
    x, ids, lookup = validate_responses(responses, ids)
    audit_ids = list(map(int, audit_ids))
    if len(set(audit_ids)) != len(audit_ids) or not set(audit_ids) <= set(ids):
        raise ValueError("Invalid audit IDs")
    counts = support(labels, audit_ids)
    rows = []
    known = [i for i in audit_ids if labels.get(i) in ["yes", "no"]]
    positive = np.array([labels[i] == "yes" for i in known])
    for q in selection["quantiles"]:
        for chosen in q["selections"]:
            status = (
                q["status"]
                if q["status"] != "complete"
                else (
                    "complete" if counts["eligible"] else "insufficient_audit_support"
                )
            )
            row = dict(
                status=status,
                negative_quantile=q["negative_quantile"],
                budget=chosen["budget"],
                groups=list(chosen["groups"]),
                groups_used=chosen["groups_used"],
                candidate_groups=75,
                description_support=selection["description_support"],
                audit_support=counts,
                metrics=None,
                target_specific_alignment_status="pending_separate_visual_review",
            )
            if status == "complete":
                groups = chosen["groups"]
                threshold = np.asarray(q["thresholds"])[groups]
                pred = x[[lookup[i] for i in known]][:, groups] > threshold
                row["metrics"] = or_metrics(pred, positive, groups)
            rows.append(row)
    return rows


def coherence_support(groups, by_unit):
    """Separate generic fixed3-pair endpoint; never filters candidates."""
    result = []
    for unit in groups:
        record = by_unit[int(unit)]
        counts = {
            s: sum(p["semantic"] == s for p in record["pairs"])
            for s in ["yes", "no", "unclear"]
        }
        if sum(counts.values()) != 3:
            raise ValueError("Expected existing fixed3-pair record")
        result.append(
            dict(
                unit=int(unit),
                counts=counts,
                meanings=[p["shared_semantic_meaning"] for p in record["pairs"]],
                wholebank_index=record["index"],
                reviewer=record["reviewer"],
                target_specific_alignment=False,
            )
        )
    return result


def alignment_view_payload(scores, ids, half_ids, unit, known_peak_image_ids, role):
    """Six highest responses within the chosen half, excluding old per-unit peaks."""
    x = np.asarray(scores)
    pooled = mean_top4(x)
    _, ids, lookup = validate_responses(pooled, ids)
    if role not in ["description", "audit"]:
        raise ValueError("Unknown evidence role")
    if int(unit) not in UNITS:
        raise ValueError("Only native0..74 eligible")
    peaks = set(map(int, known_peak_image_ids))
    eligible = [int(i) for i in half_ids if int(i) not in peaks]
    if not set(eligible) <= set(ids) or len(set(eligible)) != len(eligible):
        raise ValueError("Invalid half IDs")
    chosen = sorted(eligible, key=lambda iid: (-float(pooled[lookup[iid], unit]), iid))[
        :6
    ]
    views = []
    for iid in chosen:
        values = x[lookup[iid], :, unit]
        top = sorted(range(196), key=lambda p: (-float(values[p]), p))[:4]
        views.append(
            dict(
                image_id=iid,
                response=float(pooled[lookup[iid], unit]),
                positive_top4=[
                    dict(patch=p, row=p // 14, column=p % 14, score=float(values[p]))
                    for p in top
                    if values[p] > 0
                ],
            )
        )
    return dict(
        role=role,
        unit=int(unit),
        excluded_known_peak_ids=sorted(peaks),
        views=views,
        target_definition_fixed=True,
        alignment_judgment="pending",
        per_view_states=sorted(ALIGNMENT_STATES),
    )


def verify_artifact_gates(root, selection_path=None):
    """No production audit outputs without root freeze, complete label lock and selection."""
    root = Path(root)
    protocol_path = root / "FROZEN_PROTOCOL.json"
    protocol = read_verified(protocol_path)
    if protocol.get("status") not in [
        "FROZEN",
        "ROOT_FROZEN_BEFORE_NEW_LABELS_AND_OUTCOMES",
    ]:
        raise ValueError("Analytical protocol not frozen")
    proposal = root / "PROPOSED_PROTOCOL.json"
    if protocol.get("proposal_sha256") != file_sha(proposal):
        raise ValueError("Frozen proposal provenance mismatch")
    for name in ["categories.json", "image_manifest.json", "REVIEW_PROTOCOL.json"]:
        read_verified(root / name)
    lock_path = root / "LABEL_LOCK.json"
    lock = read_verified(lock_path)
    if (
        lock.get("protocol_sha256") != file_sha(protocol_path)
        or not lock.get("complete")
        or lock.get("planned_images") != 400
        or lock.get("planned_categories") != 75
    ):
        raise ValueError("Complete400x75 independent presence lock required")
    for item in lock.get("files", []):
        p = Path(item["path"])
        p = p if p.is_absolute() else root / p
        if file_sha(p) != item["sha256"]:
            raise ValueError("Label/source lock hash mismatch")
    if not lock.get("files"):
        raise ValueError("Label/source evidence hashes required")
    if selection_path is not None:
        path = Path(selection_path)
        selection = read_verified(path)
        if (
            not selection.get("description_only")
            or not selection.get("complete")
            or selection.get("protocol_sha256") != file_sha(protocol_path)
            or selection.get("label_lock_sha256") != file_sha(lock_path)
        ):
            raise ValueError("Description-only selection lock required")
    return protocol, lock


def verify_audit_alignment_link(root, description_alignment_path, category_key, unit):
    """Audit semantic view review requires this exact category/group description lock."""
    path = Path(description_alignment_path)
    lock = read_verified(path)
    if not lock.get("locked") or lock.get("role") != "description":
        raise ValueError("Description alignment is not locked")
    links = lock.get("links", [])
    if not any(
        x.get("category_key") == category_key
        and x.get("unit") == unit
        and x.get("locked")
        for x in links
    ):
        raise ValueError(
            "Exact category/group description alignment must be locked first"
        )
    return lock
