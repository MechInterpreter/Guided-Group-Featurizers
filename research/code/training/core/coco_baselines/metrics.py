"""Annotation-grounded proxies, never a claim of latent-manifold ground truth."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from .storage import array_sha


def ap(scores, positive, valid=None, weights=None):
    """Exact non-interpolated AP, including tied scores and ignored patches."""
    scores = np.asarray(scores)
    y = np.asarray(positive, bool)
    valid = np.ones_like(y, bool) if valid is None else np.asarray(valid, bool)
    w = (
        np.ones(len(y), np.float64)
        if weights is None
        else np.asarray(weights, np.float64)
    )
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    if not len(s):
        return None
    tp = np.cumsum((y & valid)[order] * w[order])
    known = np.cumsum(valid[order] * w[order])
    if tp[-1] == 0 or known[-1] == tp[-1]:
        return None
    precision = np.divide(
        tp[ends], known[ends], out=np.zeros(len(ends)), where=known[ends] > 0
    )
    return float(np.sum(np.diff(np.r_[0.0, tp[ends]]) * precision) / tp[-1])


def ap_matrix(heat, y, valid):
    """Sort once per group; evaluate all category masks together."""
    out = np.full((heat.shape[1], y.shape[1]), np.nan)
    for g in range(heat.shape[1]):
        order = np.argsort(-heat[:, g], kind="stable")
        s = heat[order, g]
        ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
        tp = np.cumsum((y & valid)[order], axis=0, dtype=np.float64)
        known = np.cumsum(valid[order], axis=0, dtype=np.float64)
        precision = np.divide(
            tp[ends], known[ends], out=np.zeros_like(tp[ends]), where=known[ends] > 0
        )
        numerator = (
            np.diff(np.vstack([np.zeros((1, y.shape[1])), tp[ends]]), axis=0)
            * precision
        ).sum(0)
        np.divide(
            numerator, tp[-1], out=out[g], where=(tp[-1] > 0) & (known[-1] > tp[-1])
        )
    return out


def binary_labels(audit, cfg):
    f = audit["fraction"].astype(np.float32) / 255
    y = f >= cfg["positive_fraction"]
    valid = ((f >= cfg["positive_fraction"]) | (f <= cfg["negative_fraction"])) & (
        audit["crowd"][:, None] <= cfg["negative_fraction"]
    )
    return y, valid


def image_counts(positive, valid, image_indices):
    return len(np.unique(image_indices[positive & valid])), len(
        np.unique(image_indices[~positive & valid])
    )


def f1_threshold(scores, y, valid):
    order = np.argsort(-scores, kind="stable")
    s = scores[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:]), len(s) - 1]
    tp = np.cumsum((y & valid)[order])
    count = np.cumsum(valid[order])
    total = int((y & valid).sum())
    f1 = np.divide(
        2 * tp[ends],
        count[ends] + total,
        out=np.zeros(len(ends), float),
        where=count[ends] + total > 0,
    )
    return float(s[ends[int(np.argmax(f1))]])


def detection(scores, y, valid, threshold):
    pred = scores >= threshold
    tp = int((pred & y & valid).sum())
    fp = int((pred & ~y & valid).sum())
    fn = int((~pred & y & valid).sum())
    tn = int((~pred & ~y & valid).sum())
    return dict(
        ap=ap(scores, y, valid),
        f1=2 * tp / max(2 * tp + fp + fn, 1),
        iou=tp / max(tp + fp + fn, 1),
        precision=tp / max(tp + fp, 1),
        recall=tp / max(tp + fn, 1),
        false_positive_rate=fp / max(fp + tn, 1),
        prevalence=float((y & valid).sum() / max(valid.sum(), 1)),
        valid_patches=int(valid.sum()),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def capture(target, contribution):
    """Explained object-contrast energy, relative to predicting no contrast."""
    denom = float(np.sum(np.asarray(target, np.float64) ** 2))
    if denom <= 1e-12:
        return None
    return float(
        1 - np.sum((np.asarray(target, np.float64) - contribution) ** 2) / denom
    )


def contrast_mapping(target, dz, atoms, max_groups):
    """Greedy annotation-audit subset selection; no refitting of decoder/code."""
    if len(target) == 0:
        return None
    residual = target.astype(np.float64).copy()
    chosen = []
    scores = []
    norms = np.einsum(
        "mgr,grs,mgs->g",
        dz.astype(float),
        np.einsum("grd,gsd->grs", atoms.astype(float), atoms.astype(float)),
        dz.astype(float),
    )
    denominator = float((residual**2).sum())
    if denominator <= 1e-12:
        return None
    for k in range(min(max_groups, atoms.shape[0])):
        projection = np.einsum("md,grd->mgr", residual, atoms, optimize=True)
        improvement = 2 * np.einsum("mgr,mgr->g", dz, projection) - norms
        improvement[chosen] = -np.inf
        g = int(np.argmax(improvement))
        if k and improvement[g] <= 0:
            break
        chosen.append(g)
        residual -= dz[:, g] @ atoms[g]
        scores.append(float(1 - (residual**2).sum() / denominator))
    return dict(groups=chosen, calibration_capture=scores)


@torch.no_grad()
def audit_arrays(model, x, masks, ids, cfg, device):
    """All patches for contrasts; label-blind random patch subset for detection."""
    model.eval()
    atoms = model.atoms().cpu().numpy().astype(np.float32)
    heat = []
    frac = []
    crowd = []
    image_index = []
    dx = []
    dz = []
    objects = []
    previews = []
    skipped = dict(
        crowd_instances_already_excluded=True,
        too_little_foreground=0,
        too_little_background=0,
    )
    for lo in range(0, len(x), cfg["audit_image_batch"]):
        xb = torch.as_tensor(
            x[lo : lo + cfg["audit_image_batch"]].reshape(-1, x.shape[-1]),
            device=device,
        )
        z = (
            model.encode(xb)
            .cpu()
            .numpy()
            .reshape(-1, 196, model.n_groups, model.group_size)
        )
        for j, zz in enumerate(z):
            i = lo + j
            mm = masks[i]
            pick = np.random.default_rng(cfg["patch_sample_seed"] + int(ids[i])).choice(
                196, min(cfg["detection_patches"], 196), replace=False
            )
            norm = np.linalg.norm(zz, axis=-1)
            heat.append(norm[pick])
            frac.append(mm["fraction"][pick])
            crowd.append(mm["crowd"][pick])
            image_index.extend([i] * len(pick))
            if i < cfg["preview_images"]:
                previews.append(
                    dict(image_id=ids[i], heat=norm, fraction=mm["fraction"])
                )
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
                weights = fg / fg.sum() - bg / bg.sum()
                dx.append(weights @ x[i])
                dz.append(np.einsum("p,pgr->gr", weights, zz))
                objects.append(
                    dict(
                        image_id=int(ids[i]),
                        image_index=i,
                        annotation_id=obj["annotation_id"],
                        category=obj["category"],
                        foreground_mass=float(fg.sum()),
                        background_patches=int(bg.sum()),
                    )
                )
    result = dict(
        heat=np.concatenate(heat).astype(np.float32),
        fraction=np.concatenate(frac),
        crowd=np.concatenate(crowd),
        image_index=np.array(image_index),
        image_ids=ids,
        dx=np.array(dx, dtype=np.float32).reshape(-1, model.d),
        dz=np.array(dz, dtype=np.float32).reshape(-1, model.n_groups, model.group_size),
        objects=objects,
        atoms=atoms,
        skipped=skipped,
        previews=previews,
    )
    result["input_sha"] = array_sha(x)
    return result


def calibrate(audit, cfg, categories):
    """Only this split may choose groups, thresholds, or evaluation probes."""
    y, valid = binary_labels(audit, cfg)
    matrix = ap_matrix(audit["heat"], y, valid)
    mappings = []
    object_cats = np.array([o["category"] for o in audit["objects"]])
    for c, category in enumerate(categories):
        positive_images, negative_images = image_counts(
            y[:, c], valid[:, c], audit["image_index"]
        )
        objs = np.flatnonzero(object_cats == c)
        item = dict(
            category_index=c,
            category=category["name"],
            category_id=category["id"],
            calibration_positive_images=positive_images,
            calibration_negative_images=negative_images,
            calibration_objects=len(objs),
            detector=None,
            contrast=None,
        )
        if "category_indices" in cfg and c not in cfg["category_indices"]:
            item["scope_exclusion"] = (
                "Outside the category set frozen by the pretraining coverage audit"
            )
            mappings.append(item)
            continue
        if (
            positive_images >= cfg["min_category_images"]
            and negative_images >= cfg["min_category_images"]
            and np.isfinite(matrix[:, c]).any()
        ):
            g = int(np.nanargmax(matrix[:, c]))
            threshold = f1_threshold(audit["heat"][:, g], y[:, c], valid[:, c])
            top = np.argsort(-np.nan_to_num(matrix[:, c], nan=-1))[
                : cfg["max_audit_groups"]
            ].tolist()
            X = audit["heat"][:, top]
            mu = X.mean(0)
            sigma = X.std(0).clip(1e-6)
            probe = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=500,
                random_state=cfg["patch_sample_seed"],
            )
            probe.fit(((X - mu) / sigma)[valid[:, c]], y[valid[:, c], c])
            item["detector"] = dict(
                group=g,
                threshold=threshold,
                calibration_ap=float(matrix[g, c]),
                groups=top,
                mu=mu.tolist(),
                sigma=sigma.tolist(),
                coefficient=probe.coef_[0].tolist(),
                intercept=float(probe.intercept_[0]),
            )
        if (
            len(objs) >= cfg["min_category_images"]
            and len({audit["objects"][i]["image_id"] for i in objs})
            >= cfg["min_category_images"]
        ):
            item["contrast"] = contrast_mapping(
                audit["dx"][objs],
                audit["dz"][objs],
                audit["atoms"],
                cfg["max_audit_groups"],
            )
        mappings.append(item)
    return dict(
        mappings=mappings,
        calibration_group_ap=matrix,
        groups=model_groups(audit),
        rule="Group/threshold/probe/subset chosen on calibration only. Test never changes any mapping.",
        label_use="Post-training supervised audit only; annotations never choose model weights or training hyperparameters.",
    )


def model_groups(audit):
    return int(audit["atoms"].shape[0])


def interval(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return (
        None
        if not len(values)
        else [float(v) for v in np.quantile(values, [0.025, 0.975])]
    )


def contrast_report(audit, indices, mapping, cfg, seed):
    target = audit["dx"][indices].astype(np.float64)
    dz = audit["dz"][indices]
    D = audit["atoms"]
    recon_all = np.einsum("mgr,grd->md", dz, D, optimize=True)
    sums = np.zeros_like(target)
    captures = []
    numerators = []
    denominator = (target**2).sum(1)
    for g in mapping["groups"]:
        sums += dz[:, g] @ D[g]
        error = ((target - sums) ** 2).sum(1)
        numerators.append(error)
        captures.append(capture(target, sums))
    if not captures:
        return None
    image_ids = np.array([audit["objects"][i]["image_id"] for i in indices])
    unique, inv = np.unique(image_ids, return_inverse=True)
    den = np.bincount(inv, weights=denominator)
    errors = [np.bincount(inv, weights=e) for e in numerators]
    rng = np.random.default_rng(seed)
    single = []
    multi = []
    gains = []
    for _ in range(cfg["bootstrap"]):
        take = rng.integers(len(unique), size=len(unique))
        d = den[take].sum()
        if d > 1e-12:
            a = float(1 - errors[0][take].sum() / d)
            b = float(1 - errors[-1][take].sum() / d)
            single.append(a)
            multi.append(b)
            gains.append(b - a)
    k80 = next(
        (j + 1 for j, v in enumerate(captures) if v >= cfg["capture_threshold"]), None
    )
    all_capture = capture(target, recon_all)
    return dict(
        single_group_capture=captures[0],
        multi_group_capture=captures[-1],
        capture_by_prefix=captures,
        all_group_capture=all_capture,
        groups=mapping["groups"],
        k_for_threshold=k80,
        threshold=cfg["capture_threshold"],
        single_ci95=interval(single),
        multi_ci95=interval(multi),
        multi_minus_single=captures[-1] - captures[0],
        gain_ci95=interval(gains),
        object_count=len(indices),
        image_count=len(unique),
        distributed_object_contrast=bool(
            all_capture is not None
            and all_capture >= cfg["capture_threshold"]
            and captures[0] < cfg["capture_threshold"] <= captures[-1]
            and len(mapping["groups"]) > 1
        ),
        interpretation="Native group contributions reconstruct an annotation-defined object-versus-rest contrast. Background/context and multiple attributes can contribute; this is not a true feature-splitting rate.",
    )


def evaluate(audit, frozen, cfg, seed):
    y, valid = binary_labels(audit, cfg)
    rows = []
    cats = np.array([o["category"] for o in audit["objects"]])
    for item in frozen["mappings"]:
        c = item["category_index"]
        row = {k: v for k, v in item.items() if k not in ("detector", "contrast")}
        p, n = image_counts(y[:, c], valid[:, c], audit["image_index"])
        row.update(
            test_positive_images=p,
            test_negative_images=n,
            detection=None,
            contrast=None,
            exclusions=[],
        )
        if item["detector"] is not None and min(p, n) >= cfg["min_category_images"]:
            m = item["detector"]
            score = audit["heat"][:, m["group"]]
            single = detection(score, y[:, c], valid[:, c], m["threshold"])
            X = (audit["heat"][:, m["groups"]] - m["mu"]) / m["sigma"]
            multi_score = X @ np.array(m["coefficient"]) + m["intercept"]
            multi_ap = ap(multi_score, y[:, c], valid[:, c])
            rng = np.random.default_rng(seed + c)
            boot = []
            gains = []
            image_ids = audit["image_index"]
            for _ in range(cfg["bootstrap"]):
                counts = np.bincount(
                    rng.integers(len(audit["image_ids"]), size=len(audit["image_ids"])),
                    minlength=len(audit["image_ids"]),
                )
                weights = counts[image_ids]
                a = ap(score, y[:, c], valid[:, c], weights)
                b = ap(multi_score, y[:, c], valid[:, c], weights)
                if a is not None:
                    boot.append(a)
                if a is not None and b is not None:
                    gains.append(b - a)
            row["detection"] = dict(
                single,
                group=m["group"],
                threshold=m["threshold"],
                ap_ci95=interval(boot),
                multi_group_probe_ap=multi_ap,
                probe_groups=m["groups"],
                probe_gain=multi_ap - single["ap"],
                probe_gain_ci95=interval(gains),
            )
            # Mask-boundary sensitivity uses the same frozen group and threshold.
            row["detection"]["mask_sensitivity"] = {}
            for pos in (0.25, 0.75):
                yy, vv = binary_labels(audit, dict(cfg, positive_fraction=pos))
                row["detection"]["mask_sensitivity"][str(pos)] = detection(
                    score, yy[:, c], vv[:, c], m["threshold"]
                )
        else:
            row["exclusions"].append(
                "Insufficient positive/negative images for reliable detector audit"
            )
        indices = np.flatnonzero(cats == c)
        n_images = len({audit["objects"][i]["image_id"] for i in indices})
        if item["contrast"] is not None and n_images >= cfg["min_category_images"]:
            row["contrast"] = contrast_report(
                audit, indices, item["contrast"], cfg, seed + c
            )
            cg = item["contrast"]["groups"][0]
            row["contrast"]["single_group_id"] = cg
            row["contrast"]["chosen_group_object_ap"] = ap(
                audit["heat"][:, cg], y[:, c], valid[:, c]
            )
            if item["detector"] is not None:
                dg = item["detector"]["group"]
                row["contrast"]["detector_group_id"] = dg
                row["contrast"]["detector_group_contrast_capture"] = capture(
                    audit["dx"][indices], audit["dz"][indices, dg] @ audit["atoms"][dg]
                )
                row["contrast"]["same_group_as_detector"] = bool(cg == dg)
        else:
            row["exclusions"].append(
                "Insufficient visible-object images for contrast audit"
            )
        rows.append(row)
    return rows
