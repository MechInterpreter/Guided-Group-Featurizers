"""Training-only group anchors and fixed-reference held-out preservation checks.

Gram similarity alone permits rotations in activation space. Direct decoded
contributions anchor that orientation, including the actual sparse gate.
"""

import numpy as np
import torch

from coco_baselines.storage import array_sha, identity


def build(model, train, targets, cfg, selection, confirmation):
    """No validation labels decide which groups to protect or which patches replay."""
    entries = []
    skipped = []
    rng = np.random.default_rng(cfg["anchor_seed"])
    for t in targets:
        for role, field, group_field, threshold in (
            (
                "compact_capture",
                "training_capture",
                "contrast_group",
                cfg["protect_capture_threshold"],
            ),
            (
                "strong_detector",
                "training_ap",
                "detector_group",
                cfg["protect_ap_threshold"],
            ),
        ):
            value = t[field]
            if value is None or value < threshold:
                continue
            c = t["category_index"]
            samples = []
            locations = []
            # Sample whole images first so large objects cannot dominate replay.
            for i in rng.permutation(len(train["ids"])):
                mask = train["masks"][i]
                ix = np.flatnonzero(
                    (
                        mask["fraction"][:, c].astype(float) / 255
                        >= cfg["positive_fraction"]
                    )
                    & (mask["crowd"] <= cfg["negative_fraction"])
                )
                if not len(ix):
                    continue
                ix = rng.choice(
                    ix, min(len(ix), cfg["anchor_patches_per_image"]), replace=False
                )
                samples.extend(train["x"][i, ix])
                locations.extend((int(train["ids"][i]), int(p)) for p in ix)
                if len(samples) >= cfg["anchor_examples_per_group"]:
                    break
            if not samples:
                skipped.append(
                    dict(
                        category=t["category"],
                        role=role,
                        reason="no_non_crowd_object_patches",
                    )
                )
                continue
            x = np.asarray(samples[: cfg["anchor_examples_per_group"]], np.float32)
            g = int(t[group_field])
            model.eval()
            with torch.no_grad():
                xx = torch.as_tensor(x, device=next(model.parameters()).device)
                z = model.encode(xx).reshape(-1, model.n_groups, model.group_size)[:, g]
                D = model.decoder_atoms().reshape(
                    model.n_groups, model.group_size, model.d
                )[g]
                contribution = (z @ D).cpu().numpy()
            if (
                float(np.sum(contribution.astype(float) ** 2))
                <= cfg["anchor_energy_floor"]
            ):
                skipped.append(
                    dict(
                        category=t["category"],
                        role=role,
                        group=g,
                        reason="near_zero_saved_contribution",
                    )
                )
                continue
            entries.append(
                dict(
                    category=t["category"],
                    category_index=c,
                    role=role,
                    group=g,
                    training_value=float(value),
                    x=x,
                    contribution=contribution,
                    locations=locations[: len(x)],
                    sample_sha=array_sha(x),
                    contribution_sha=array_sha(contribution),
                )
            )
    book = dict(
        entries=entries,
        skipped=skipped,
        selection_reference=selection,
        confirmation_reference=confirmation,
        source="Original native model; labels and replay samples from training only",
        promotion_policy="Fixed native anchors for this diagnostic; no automatic promotion",
    )
    book["id"] = identity(
        dict(
            entries=[
                {k: v for k, v in e.items() if k not in ("x", "contribution")}
                for e in entries
            ],
            selection=selection["categories"],
            confirmation=confirmation["categories"],
        )
    )
    return book


def summary(book):
    grouped = {}
    for e in book["entries"]:
        grouped.setdefault(e["group"], set()).add(e["category"])
    return dict(
        id=book["id"],
        source=book["source"],
        promotion_policy=book["promotion_policy"],
        entries=[
            {k: v for k, v in e.items() if k not in ("x", "contribution")}
            for e in book["entries"]
        ],
        skipped=book["skipped"],
        compact_capture_pairs=sum(
            e["role"] == "compact_capture" for e in book["entries"]
        ),
        strong_detector_pairs=sum(
            e["role"] == "strong_detector" for e in book["entries"]
        ),
        shared_protected_groups=[
            dict(group=g, categories=sorted(c))
            for g, c in grouped.items()
            if len(c) > 1
        ],
        empty=not bool(book["entries"]),
    )


def losses(model, book, cfg, update=0, all_entries=False):
    zero = next(model.parameters()).sum() * 0
    if not book or not book["entries"]:
        return zero, zero, []
    direct = []
    gram = []
    rows = []
    D = model.decoder_atoms().reshape(model.n_groups, model.group_size, model.d)
    # Round-robin replay caps compute without starving any protected pair.
    count = (
        len(book["entries"])
        if all_entries
        else min(cfg["anchor_pairs_per_update"], len(book["entries"]))
    )
    indices = [(update * count + j) % len(book["entries"]) for j in range(count)]
    for j in indices:
        e = book["entries"][j]
        x = torch.as_tensor(e["x"], device=D.device)
        ref = torch.as_tensor(e["contribution"], device=D.device)
        z = model.encode(x).reshape(-1, model.n_groups, model.group_size)[:, e["group"]]
        now = z @ D[e["group"]]
        energy = ref.square().sum().clamp_min(cfg["anchor_energy_floor"])
        a = (now - ref).square().sum() / energy
        # Shared saved-energy normalization retains scale; do not normalize each
        # current Gram independently, which could conceal magnitude collapse.
        b = ((now @ now.T - ref @ ref.T) / energy).square().sum()
        direct.append(a)
        gram.append(b)
        rows.append(
            dict(
                category=e["category"],
                group=e["group"],
                role=e["role"],
                direct_drift=float(a.detach()),
                gram_drift=float(b.detach()),
                response_energy_ratio=float((now.square().sum() / energy).detach()),
                active_fraction=float((z.norm(dim=-1) > 1e-6).float().mean().detach()),
            )
        )
    return torch.stack(direct).mean(), torch.stack(gram).mean(), rows


def guard(value, book, cfg, split="selection"):
    """Compare the SAME original group against native baseline; no target exemption."""
    if not book:
        return dict(passed=True, checks=[], reasons=[], empty=True)
    ref = {r["category_index"]: r for r in book[split + "_reference"]["categories"]}
    current = {
        r["category_index"]: r
        for r in value.get("protection_categories", value["categories"])
    }
    checks = []
    reasons = []
    for e in book["entries"]:
        field = "capture" if e["role"] == "compact_capture" else "ap"
        group_field = "contrast_group" if field == "capture" else "detector_group"
        r = ref.get(e["category_index"], {})
        c = current.get(e["category_index"], {})
        a, b = r.get(field), c.get(field)
        reason = None
        status = "measured_preserved"
        if a is None or not np.isfinite(a):
            reason = "protected_pair_insufficient_" + split + "_evidence"
            status = "setup_coverage_missing"
        elif b is None or not np.isfinite(b):
            reason = "protected_pair_insufficient_" + split + "_evidence"
            status = "candidate_score_unavailable"
        elif r.get(group_field) != e["group"] or c.get(group_field) != e["group"]:
            reason = "protected_group_identity_changed"
        elif b < a - cfg["protected_metric_tolerance"]:
            reason = "protected_" + field + "_declined_from_native"
        if reason and status == "measured_preserved":
            status = (
                "identity_mismatch" if "identity" in reason else "measured_deteriorated"
            )
        checks.append(
            dict(
                category=e["category"],
                group=e["group"],
                role=e["role"],
                metric=field,
                baseline=a,
                current=b,
                delta=None if a is None or b is None else b - a,
                tolerance=cfg["protected_metric_tolerance"],
                passed=reason is None,
                reason=reason,
                status=status,
            )
        )
        if reason:
            reasons.append(reason)
    return dict(
        passed=not reasons,
        checks=checks,
        reasons=sorted(set(reasons)),
        empty=not bool(book["entries"]),
    )


def gradient_diagnostics(model, terms):
    """Separate loss gradient strength and conflict; correlations are not causality."""
    params = [p for p in model.parameters() if p.requires_grad]
    vectors = {}
    norms = {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        v = torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).reshape(-1)
                for p, g in zip(params, grads)
            ]
        )
        vectors[name] = v
        norms[name] = float(v.norm().detach())
    cos = {}
    for name in ("contribution_anchor", "gram_anchor"):
        for other in ("native", "capture", "detection"):
            a, b = vectors[name], vectors[other]
            den = a.norm() * b.norm()
            cos[name + "_vs_" + other] = (
                None if float(den) < 1e-12 else float((a @ b / den).detach())
            )
    return dict(
        weighted_loss_gradient_norms=norms,
        gradient_cosines=cos,
        interpretation="Negative cosine means opposing local gradient directions; it does not establish which objective is correct.",
    )
