"""Explicit supervised objective, reversible edits, and diagnostic decision gates."""

import copy
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

from coco_baselines import metrics, training
from . import protection


def new_model(method, base_cfg, weights, device):
    d = (
        weights["B_raw"].shape[1]
        if method == "grassmannian"
        else weights["W_dec"].shape[1]
    )
    m = training.make_model(method, base_cfg, d).to(device)
    m.load_state_dict(weights)
    return m


def choose_targets(audit, cfg, categories):
    """Training masks alone choose two mappings, not test or validation labels."""
    y, valid = metrics.binary_labels(audit, cfg)
    ap = metrics.ap_matrix(audit["heat"], y, valid)
    cats = np.array([o["category"] for o in audit["objects"]])
    targets = []
    for c, category in enumerate(categories):
        if "category_indices" in cfg and c not in cfg["category_indices"]:
            continue
        idx = np.flatnonzero(cats == c)
        p, n = metrics.image_counts(y[:, c], valid[:, c], audit["image_index"])
        if (
            min(p, n) < cfg["target_min_images"]
            or len({audit["objects"][i]["image_id"] for i in idx})
            < cfg["target_min_images"]
        ):
            continue
        if not np.isfinite(ap[:, c]).any():
            continue
        chosen = metrics.contrast_mapping(
            audit["dx"][idx], audit["dz"][idx], audit["atoms"], 1
        )
        if not chosen:
            continue
        g = int(np.nanargmax(ap[:, c]))
        h = audit["heat"][:, g]
        targets.append(
            dict(
                category_index=c,
                category=category["name"],
                detector_group=g,
                contrast_group=chosen["groups"][0],
                heat_mean=float(h.mean()),
                heat_scale=max(float(h.std()), 1e-3),
                training_ap=float(ap[g, c]),
                training_capture=chosen["calibration_capture"][0],
                training_positive_images=p,
            )
        )
    if not targets:
        raise ValueError(
            "No categories have enough supervision; increase image count or use smoke only for plumbing"
        )
    return targets


def object_weights(masks, target_categories, cfg):
    """All visible instances, using the same contrast convention as the audit."""
    weights = []
    image = []
    cats = []
    for i, mask in enumerate(masks):
        known = mask["crowd"] <= cfg["negative_fraction"]
        for obj in mask["instances"]:
            c = obj["category"]
            if c not in target_categories:
                continue
            fg = np.asarray(obj["fraction"], float) * known
            bg = (
                mask["fraction"][:, c].astype(float) / 255 <= cfg["negative_fraction"]
            ) & known
            if (
                fg.sum() < cfg["min_object_patch_mass"]
                or bg.sum() < cfg["min_background_patches"]
            ):
                continue
            weights.append(fg / fg.sum() - bg / bg.sum())
            image.append(i)
            cats.append(c)
    return (
        np.asarray(weights, np.float32).reshape(-1, 196),
        np.array(image, int),
        np.array(cats, int),
    )


def supervised_loss(model, x, masks, targets, head, cfg, return_details=False):
    """Native gated contributions; one-group contrast + balanced mask detection."""
    ni, npats, d = x.shape
    z = model.encode(x.reshape(-1, d)).reshape(
        ni, npats, model.n_groups, model.group_size
    )
    # BSF.atoms() is an intentionally detached visualization helper. Use the
    # differentiable decoder here so this supervision also updates directions.
    D = model.decoder_atoms().reshape(model.n_groups, model.group_size, model.d)
    cap_losses = []
    det_losses = []
    objects = 0
    details = (
        {str(t["category_index"]): dict(capture=None, objects=0) for t in targets}
        if return_details
        else None
    )
    cats = [t["category_index"] for t in targets]
    ww, ii, cc = object_weights(masks, cats, cfg)
    if len(ii):
        w = torch.as_tensor(ww, device=x.device)
        ix = torch.as_tensor(ii, device=x.device)
        dx = torch.einsum("mp,mpd->md", w, x[ix])
        dz = torch.einsum("mp,mpgr->mgr", w, z[ix])
        for t in targets:
            take = torch.as_tensor(cc == t["category_index"], device=x.device)
            if not bool(take.any()):
                continue
            g = t["contrast_group"]
            prediction = dz[take, g] @ D[g]
            den = dx[take].square().sum(-1).clamp_min(1e-6)
            value = ((dx[take] - prediction).square().sum(-1) / den).mean()
            cap_losses.append(value)
            if return_details:
                details[str(t["category_index"])].update(
                    capture=value, objects=int(take.sum())
                )
        objects = len(ii)
    frac = torch.as_tensor(
        np.stack([m["fraction"] for m in masks]).astype(np.float32) / 255,
        device=x.device,
    )
    crowd = torch.as_tensor(np.stack([m["crowd"] for m in masks]), device=x.device)
    for j, t in enumerate(targets):
        f = frac[:, :, t["category_index"]]
        pos = f >= cfg["positive_fraction"]
        neg = f <= cfg["negative_fraction"]
        valid = (pos | neg) & (crowd <= cfg["negative_fraction"])
        pos = pos & valid
        neg = neg & valid
        h = (z[:, :, t["detector_group"]].norm(dim=-1) - t["heat_mean"]) / t[
            "heat_scale"
        ]
        logits = F.softplus(head[j, 0]) * h + head[j, 1]
        terms = []
        if bool(pos.any()):
            terms.append(F.softplus(-logits[pos]).mean())
        if bool(neg.any()):
            terms.append(F.softplus(logits[neg]).mean())
        if terms:
            det_losses.append(torch.stack(terms).mean())
        if return_details:
            details[str(t["category_index"])].update(
                detection=torch.stack(terms).mean() if terms else z.sum() * 0,
                positive_patches=int(pos.sum()),
                negative_patches=int(neg.sum()),
            )
    zero = z.sum() * 0
    cap = torch.stack(cap_losses).mean() if cap_losses else zero
    det = torch.stack(det_losses).mean() if det_losses else zero
    return (cap, det, objects, details) if return_details else (cap, det, objects)


def validation(
    model, pack, targets, cfg, device, with_arrays=False, protection_pack=None
):
    audit = metrics.audit_arrays(
        model, pack["x"], pack["masks"], pack["ids"], cfg, device
    )
    out = category_scores(audit, targets, cfg)
    out["reconstruction"] = training.measure(
        model, pack["x"].reshape(-1, model.d), device, cfg["batch"]
    )
    if with_arrays:
        out["_audit"] = audit
    if protection_pack is not None:
        pa = metrics.audit_arrays(
            model,
            protection_pack["x"],
            protection_pack["masks"],
            protection_pack["ids"],
            cfg,
            device,
        )
        out["protection_categories"] = category_scores(pa, targets, cfg)["categories"]
    return out


def category_scores(audit, targets, cfg):
    y, valid = metrics.binary_labels(audit, cfg)
    oc = np.array([o["category"] for o in audit["objects"]])
    rows = []
    for t in targets:
        c = t["category_index"]
        idx = np.flatnonzero(oc == c)
        p, n = metrics.image_counts(y[:, c], valid[:, c], audit["image_index"])
        cap = None
        ap = None
        if (
            len({audit["objects"][i]["image_id"] for i in idx})
            >= cfg["decision_min_images"]
        ):
            cap = metrics.capture(
                audit["dx"][idx],
                audit["dz"][idx, t["contrast_group"]]
                @ audit["atoms"][t["contrast_group"]],
            )
        if min(p, n) >= cfg["decision_min_images"]:
            ap = metrics.ap(audit["heat"][:, t["detector_group"]], y[:, c], valid[:, c])
        known_positive = y[:, c] & valid[:, c]
        h = audit["heat"][:, t["detector_group"]]
        support = (
            float((h[known_positive] > 1e-6).mean()) if known_positive.any() else None
        )
        rows.append(
            dict(
                category=t["category"],
                category_index=c,
                capture=cap,
                ap=ap,
                object_count=len(idx),
                positive_images=p,
                negative_images=n,
                contrast_group=t["contrast_group"],
                detector_group=t["detector_group"],
                detector_active_fraction_on_objects=support,
            )
        )
    caps = [r["capture"] for r in rows if r["capture"] is not None]
    aps = [r["ap"] for r in rows if r["ap"] is not None]
    out = dict(
        categories=rows,
        macro_capture=float(np.mean(caps)) if caps else None,
        macro_ap=float(np.mean(aps)) if aps else None,
    )
    out["score"] = (
        None
        if not caps or not aps
        else out["macro_capture"] + cfg["score_ap_weight"] * out["macro_ap"]
    )
    return out


def fit(
    store,
    key,
    method,
    base_cfg,
    initial,
    train_pack,
    val_pack,
    targets,
    cfg,
    updates,
    supervised,
    seed,
    device,
    max_updates=None,
    anchors=None,
    protection_pack=None,
):
    """Fresh common Adam per arm; every arm sees the same seeded minibatches."""
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    m = new_model(method, base_cfg, initial, device)
    head = torch.nn.Parameter(torch.zeros(len(targets), 2, device=device))
    opt = torch.optim.Adam(
        list(m.parameters()) + ([head] if supervised else []), lr=cfg["lr"]
    )
    rng = np.random.default_rng(seed)
    st = store.load(key)
    anchor_id = anchors["id"] if anchors is not None else None
    if st is None:
        base = validation(
            m, val_pack, targets, cfg, device, protection_pack=protection_pack
        )
        st = dict(
            updates=0,
            target=updates,
            initial_sha=training.weight_sha(initial),
            weights=training.weights(m),
            head=head.detach().cpu(),
            best=training.weights(m),
            best_score=base["score"] if supervised else -base["reconstruction"]["mse"],
            best_update=0,
            history=[dict(update=0, validation=base)],
            seconds=0.0,
            targets=targets,
            supervised=supervised,
            anchor_id=anchor_id,
            initial_mse=base["reconstruction"]["mse"],
            samples_per_update=cfg["images_per_update"] * 196,
        )
        st["history"][0]["protection"] = protection.guard(base, anchors, cfg)
        if not st["history"][0]["protection"]["passed"]:
            st["best_score"] = None
            st["best_valid"] = False
        else:
            st["best_valid"] = True
    else:
        if (
            st["target"] != updates
            or st["initial_sha"] != training.weight_sha(initial)
            or st["targets"] != targets
            or st["supervised"] != supervised
        ):
            raise ValueError("Fit identity changed")
        if st.get("anchor_id") != anchor_id:
            raise ValueError("Protected anchor identity changed")
        m.load_state_dict(st["weights"])
        head.data.copy_(st["head"].to(device))
        opt.load_state_dict(st["optimizer"])
        rng.bit_generator.state = st["sample_rng"]
        torch.set_rng_state(st["torch_rng"])
        if device.startswith("cuda") and st["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(st["cuda_rng"])
    limit = updates if max_updates is None else min(updates, max_updates)
    print(key, "resume", st["updates"], "/", updates, flush=True)
    store.event(
        "fit_start",
        key=key,
        updates=st["updates"],
        target=updates,
        anchor_id=anchor_id,
        protected_pairs=0 if anchors is None else len(anchors["entries"]),
    )

    def save(drive=True):
        st.update(
            weights=training.weights(m),
            head=head.detach().cpu().clone(),
            optimizer=copy.deepcopy(opt.state_dict()),
            sample_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else None,
        )
        store.save(key, st, drive)

    try:
        while st["updates"] < limit:
            idx = rng.integers(len(train_pack["x"]), size=cfg["images_per_update"])
            x = torch.as_tensor(train_pack["x"][idx], device=device)
            m.train()
            started = time.monotonic()
            xx = x.reshape(-1, m.d)
            noisy = (
                xx + cfg["snr"] * xx.std() * torch.randn_like(xx) if cfg["snr"] else xx
            )
            native, info = m.loss(noisy, target=xx)
            cap = det = native * 0
            objects = 0
            if supervised:
                cap, det, objects = supervised_loss(
                    m, x, [train_pack["masks"][i] for i in idx], targets, head, cfg
                )
            anchor, gram, anchor_rows = protection.losses(
                m, anchors, cfg, st["updates"]
            )
            terms = dict(
                native=native,
                capture=cfg["lambda_capture"] * cap,
                detection=cfg["lambda_detection"] * det,
                contribution_anchor=cfg["lambda_contribution_anchor"] * anchor,
                gram_anchor=cfg["lambda_gram_anchor"] * gram,
            )
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite adaptation loss")
            diagnostics = (
                protection.gradient_diagnostics(m, terms)
                if (st["updates"] + 1) % cfg["diagnostic_grad_every"] == 0
                else None
            )
            opt.zero_grad()
            loss.backward()
            encoder_grad = (
                float(
                    sum(
                        p.grad.detach().square().sum()
                        for name, p in m.named_parameters()
                        if p.grad is not None and ("enc" in name or name == "gamma")
                    ).sqrt()
                )
                if any(
                    p.grad is not None and ("enc" in name or name == "gamma")
                    for name, p in m.named_parameters()
                )
                else 0.0
            )
            decoder_grad = (
                float(
                    sum(
                        p.grad.detach().square().sum()
                        for name, p in m.named_parameters()
                        if p.grad is not None and (name == "W_dec" or name == "B_raw")
                    ).sqrt()
                )
                if any(
                    p.grad is not None and (name == "W_dec" or name == "B_raw")
                    for name, p in m.named_parameters()
                )
                else 0.0
            )
            grad = float(
                torch.nn.utils.clip_grad_norm_(
                    list(m.parameters()) + ([head] if supervised else []),
                    cfg["grad_clip"],
                )
            )
            opt.step()
            m.normalize_decoder()
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            st["seconds"] += time.monotonic() - started
            st["updates"] += 1
            u = st["updates"]
            if u % cfg["evaluate_every"] == 0 or u == updates:
                val = validation(
                    m, val_pack, targets, cfg, device, protection_pack=protection_pack
                )
                eligible = val["reconstruction"]["mse"] <= st["initial_mse"] * (
                    1 + cfg["fit_reconstruction_tolerance"]
                )
                preservation = protection.guard(val, anchors, cfg)
                eligible &= preservation["passed"]
                if method == "group_lasso":
                    eligible &= (
                        abs(
                            val["reconstruction"]["active_groups"]
                            - base_cfg["active_groups"]
                        )
                        <= base_cfg["sparsity_tolerance"] * base_cfg["active_groups"]
                    )
                score = val["score"] if supervised else -val["reconstruction"]["mse"]
                if (
                    eligible
                    and score is not None
                    and (st["best_score"] is None or score > st["best_score"])
                ):
                    st.update(
                        best=training.weights(m),
                        best_score=score,
                        best_update=u,
                        best_valid=True,
                    )
                m.eval()
                with torch.no_grad():
                    _, _, full_anchor_rows = protection.losses(
                        m, anchors, cfg, all_entries=True
                    )
                row = dict(
                    update=u,
                    validation=val,
                    checkpoint_eligible=bool(eligible),
                    native_loss=float(native.detach()),
                    capture_loss=float(cap.detach()),
                    detection_loss=float(det.detach()),
                    total_loss=float(loss.detach()),
                    gradient_norm=grad,
                    encoder_gradient_norm=encoder_grad,
                    decoder_gradient_norm=decoder_grad,
                    objects_in_batch=objects,
                    seconds_per_update=st["seconds"] / u,
                    native_loss_info=info,
                    contribution_anchor_loss=float(anchor.detach()),
                    gram_anchor_loss=float(gram.detach()),
                    anchor_responses=full_anchor_rows,
                    protection=preservation,
                    component_gradients=diagnostics,
                )
                st["history"].append(row)
                save()
                store.write("histories/" + key + ".json", st["history"])
                store.event(
                    "adaptation_checkpoint",
                    key=key,
                    update=u,
                    total=updates,
                    macro_capture=val["macro_capture"],
                    macro_ap=val["macro_ap"],
                    mse=val["reconstruction"]["mse"],
                    best_update=st["best_update"],
                    seconds_per_update=row["seconds_per_update"],
                )
            elif u % cfg["local_save_every"] == 0:
                save(False)
                print(key, u, "/", updates, flush=True)
        save()
    except BaseException as exc:
        save(False)
        try:
            save()
            store.event(
                "fit_error",
                key=key,
                update=st["updates"],
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except BaseException:
            pass
        raise
    del m, head, opt
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return st


def transform_weights(method, weights, pair, Q, rank):
    """Cross-block rotation/permutation; native model architecture stays unchanged."""
    out = {k: v.detach().cpu().clone() for k, v in weights.items()}
    idx = np.concatenate([np.arange(g * rank, (g + 1) * rank) for g in pair])
    q = torch.as_tensor(Q, dtype=torch.float32)
    if not np.allclose(Q @ Q.T, np.eye(2 * rank), atol=1e-6):
        raise ValueError("Edit is not orthogonal")
    if method in ("vanilla", "group_lasso"):
        out["W_dec"][idx] = q @ out["W_dec"][idx]
        out["W_enc"][:, idx] = out["W_enc"][:, idx] @ q.T
        out["b_enc"][idx] = q @ out["b_enc"][idx]
    elif method == "grassmannian":
        # Native tied frames re-orthogonalize each new block by QR on forward.
        frames = torch.linalg.qr(out["B_raw"])[0]
        rows = frames[list(pair)].permute(0, 2, 1).reshape(2 * rank, -1)
        changed = q @ rows
        out["B_raw"][list(pair)] = changed.reshape(2, rank, -1).permute(0, 2, 1)
    else:
        raise ValueError("Unknown native method")
    return out


def proposals(audit, targets, cfg):
    """Masks guide pair selection and which coordinate variation to consolidate."""
    D = audit["atoms"]
    rank = D.shape[1]
    oc = np.array([o["category"] for o in audit["objects"]])
    pool = []
    for t in targets:
        take = np.flatnonzero(oc == t["category_index"])
        if len(take) < cfg["target_min_images"]:
            continue
        dx = audit["dx"][take]
        dz = audit["dz"][take]
        if float(np.sum(dx.astype(float) ** 2)) <= 1e-12:
            continue
        singles = []
        for g in range(D.shape[0]):
            singles.append(metrics.capture(dx, dz[:, g] @ D[g]))
        top = np.argsort(singles)[-cfg["proposal_group_pool"] :]
        for a, b in itertools.combinations(top.tolist(), 2):
            cap = metrics.capture(dx, dz[:, a] @ D[a] + dz[:, b] @ D[b])
            gap = cap - max(singles[a], singles[b])
            pool.append(
                dict(
                    pair=[a, b],
                    category_index=t["category_index"],
                    category=t["category"],
                    pair_capture=cap,
                    best_single_capture=max(singles[a], singles[b]),
                    training_pair_gain=gap,
                )
            )
    pool.sort(key=lambda p: -p["training_pair_gain"])
    out = []
    seen = set()
    for p in pool:
        pair = p["pair"]
        marker = tuple(sorted(pair))
        if marker in seen:
            continue
        seen.add(marker)
        take = oc == p["category_index"]
        v = audit["dz"][take][:, pair].reshape(-1, 2 * rank)
        gram = v.astype(float).T @ v.astype(float) / max(len(v), 1)
        ev, V = np.linalg.eigh(gram)
        Q = V[:, np.argsort(-ev)].T
        # Orient deterministic signs; this does not use held-out labels.
        signs = np.sign(Q[np.arange(2 * rank), np.argmax(np.abs(Q), axis=1)])
        Q *= np.where(signs == 0, 1, signs)[:, None]
        out.append(
            dict(
                **p,
                kind="supervised_cross_block_rotation",
                Q=Q,
                coefficient_energy_eigenvalues=ev[::-1].tolist(),
            )
        )
        energy = np.diag(gram)
        perm = np.arange(2 * rank)
        i = int(np.argmin(energy[:rank]))
        j = rank + int(np.argmax(energy[rank:]))
        perm[i], perm[j] = perm[j], perm[i]
        out.append(
            dict(
                **p,
                kind="direction_exchange",
                Q=np.eye(2 * rank)[perm],
                exchanged_local_coordinates=[i, j],
            )
        )
        if len(out) >= cfg["candidates_per_round"]:
            break
    return out[: cfg["candidates_per_round"]]


def compare(
    candidate,
    control,
    initial,
    cfg,
    base_cfg,
    method,
    target_category=None,
    anchors=None,
    split="selection",
):
    """Every rejection gets named evidence, not one opaque combined score."""
    reasons = []
    m = candidate["reconstruction"]["mse"]
    cm = control["reconstruction"]["mse"]
    if candidate["score"] is None or control["score"] is None:
        reasons.append("insufficient_annotation_evidence")
    gain = (
        None
        if candidate["score"] is None or control["score"] is None
        else candidate["score"] - control["score"]
    )
    if gain is None or gain < cfg["min_score_gain"]:
        reasons.append("composite_gain_too_small")
    if m > cm * (1 + cfg["accept_reconstruction_tolerance"]):
        reasons.append("worse_reconstruction_than_matched_control")
    if m > initial["reconstruction"]["mse"] * (
        1 + cfg["global_reconstruction_tolerance"]
    ):
        reasons.append("worse_reconstruction_than_native_start")
    if (
        method == "group_lasso"
        and abs(
            candidate["reconstruction"]["active_groups"] - base_cfg["active_groups"]
        )
        > base_cfg["sparsity_tolerance"] * base_cfg["active_groups"]
    ):
        reasons.append("sparsity_mismatch")
    if (
        candidate["macro_capture"] is not None
        and control["macro_capture"] is not None
        and candidate["macro_capture"]
        < control["macro_capture"] - cfg["macro_tradeoff_tolerance"]
    ):
        reasons.append("macro_capture_declined")
    if (
        candidate["macro_ap"] is not None
        and control["macro_ap"] is not None
        and candidate["macro_ap"]
        < control["macro_ap"] - cfg["macro_tradeoff_tolerance"]
    ):
        reasons.append("macro_detection_declined")
    off = []
    old = {r["category_index"]: r for r in control["categories"]}
    for row in candidate["categories"]:
        c = row["category_index"]
        if c == target_category or c not in old:
            continue
        for field in ("capture", "ap"):
            a, b = row[field], old[c][field]
            if (
                a is not None
                and b is not None
                and a < b - cfg["category_preservation_tolerance"]
            ):
                off.append(dict(category=row["category"], metric=field, delta=a - b))
    if off:
        reasons.append("other_object_categories_degraded")
    preservation = protection.guard(candidate, anchors, cfg, split)
    reasons.extend(preservation["reasons"])
    return dict(
        accepted=not reasons,
        reasons=reasons,
        composite_gain=gain,
        reconstruction_relative_change=m / cm - 1,
        off_target_losses=off,
        candidate=candidate,
        control=control,
        protected_pairs=preservation,
        thresholds={
            k: cfg[k]
            for k in (
                "min_score_gain",
                "accept_reconstruction_tolerance",
                "global_reconstruction_tolerance",
                "macro_tradeoff_tolerance",
                "category_preservation_tolerance",
                "protected_metric_tolerance",
            )
        },
    )
