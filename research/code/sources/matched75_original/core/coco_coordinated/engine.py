"""Coordinated, object-supervised transactions on frozen native BSF starts.

Proposal rotations are a cheap UNGATED surrogate using already gated training
codes. Recovery, checkpoint selection and decisions use the real native gate.
No object/group assignment is chosen using validation or test observations.
"""
import copy
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import expm, schur

from coco_adaptation import core
from coco_baselines import metrics, training
from coco_baselines.storage import array_sha, identity


def _number(v):
    return v is not None and np.isfinite(v)


def _array_identity(a):
    """Historical array_sha bytes, streamed so 1 GB activation copies are avoided."""
    import hashlib
    a = np.asarray(a)
    h = hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode())
    for lo in range(0, len(a), 32):
        h.update(np.ascontiguousarray(a[lo:lo + 32]).tobytes())
    return h.hexdigest()


def _gain_threshold(cfg):
    return cfg.get('min_target_capture_gain', cfg.get('target_capture_gain', .01))


def _ap_margin(cfg):
    return -cfg['min_target_ap_gain'] if 'min_target_ap_gain' in cfg else cfg.get('target_ap_tolerance', .01)


def orthogonal_homotopy(Q, strength):
    """Interpolate a proper orthogonal edit, avoiding a nonorthogonal blend."""
    Q = np.asarray(Q, np.float64)
    n = Q.shape[0]
    if Q.shape != (n, n) or not np.allclose(Q @ Q.T, np.eye(n), atol=2e-6):
        raise ValueError('Edit is not orthogonal')
    if np.linalg.det(Q) < 0:
        raise ValueError('An identity homotopy requires a proper rotation')
    if not 0 <= strength <= 1:
        raise ValueError('Edit strength must be between zero and one')
    if strength == 0:
        return np.eye(n)
    if strength == 1:
        return Q.copy()
    T, Z = schur(Q, output='real')
    L = np.zeros_like(T)
    minus = []
    i = 0
    while i < n:
        if i + 1 < n and abs(T[i + 1, i]) > 1e-8:
            angle = np.arctan2(T[i + 1, i], T[i, i])
            L[i, i + 1], L[i + 1, i] = -angle, angle
            i += 2
        else:
            if T[i, i] < 0:
                minus.append(i)
            i += 1
    if len(minus) % 2:
        raise ValueError('Cannot form a real proper-rotation logarithm')
    for a, b in zip(minus[::2], minus[1::2]):
        L[a, b], L[b, a] = -np.pi, np.pi
    out = Z @ expm(float(strength) * L) @ Z.T
    if not np.allclose(out @ out.T, np.eye(n), atol=2e-6):
        raise FloatingPointError('Rotation interpolation failed')
    return out


def _decoded(Q, z, D, rank):
    """Receiver's decoded contribution with the surrogate gate held fixed."""
    Q = np.asarray(Q)
    return (z @ Q[:rank].T) @ (Q[:rank] @ D)


def _proper(Q):
    Q = np.asarray(Q, np.float64).copy()
    if np.linalg.det(Q) < 0:
        Q[-1] *= -1
    return Q


def _rotation_score(Q, z, D, dx, rank, other, cfg):
    prediction = _decoded(Q, z, D, rank)
    target = metrics.capture(dx, prediction)
    losses = []
    for item in other:
        rows = Q[item['local_owner'] * rank:(item['local_owner'] + 1) * rank]
        pred = (item['z'] @ rows.T) @ (rows @ D)
        now = metrics.capture(item['dx'], pred)
        if now is not None and item['capture'] is not None:
            losses.append(max(0., item['capture'] - now))
    collateral = float(np.mean(losses)) if losses else 0.
    return dict(target_capture=target, collateral_mean_capture_loss=collateral,
                objective=None if target is None else float(1 - target + cfg.get('proposal_collateral_weight', 1.) * collateral))


def _optimise_rotation(z, D, dx, rank, other, seed, cfg):
    """Small SO(2r) search, directly comparing decoded object contrasts."""
    td = torch.as_tensor(D, dtype=torch.float64)
    tz = torch.as_tensor(z, dtype=torch.float64)
    tx = torch.as_tensor(dx, dtype=torch.float64)
    q0 = torch.as_tensor(seed, dtype=torch.float64)
    B = torch.nn.Parameter(torch.zeros_like(q0))
    opt = torch.optim.Adam([B], lr=cfg.get('proposal_opt_lr', .04))
    denominator = tx.square().sum().clamp_min(1e-12)
    additional = []
    for item in other:
        additional.append((torch.as_tensor(item['z'], dtype=torch.float64),
                           torch.as_tensor(item['dx'], dtype=torch.float64),
                           item['local_owner'], 1. - item['capture']))
    best_Q = np.asarray(seed, float).copy()
    best = _rotation_score(best_Q, z, D, dx, rank, other, cfg)
    trace = []
    steps = cfg.get('proposal_steps', cfg.get('proposal_opt_steps', 40))
    for u in range(steps):
        Q = torch.matrix_exp(B - B.T) @ q0
        pred = (tz @ Q[:rank].T) @ (Q[:rank] @ td)
        target_loss = (tx - pred).square().sum() / denominator
        collateral = []
        for zz, xx, owner, reference_error in additional:
            rows = Q[owner * rank:(owner + 1) * rank]
            pp = (zz @ rows.T) @ (rows @ td)
            error = (xx - pp).square().sum() / xx.square().sum().clamp_min(1e-12)
            collateral.append(F.relu(error - reference_error))
        extra = torch.stack(collateral).mean() if collateral else target_loss * 0
        loss = target_loss + cfg.get('proposal_collateral_weight', 1.) * extra
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite decoded rotation objective')
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([B], 10.)
        opt.step()
        with torch.no_grad():
            candidate = (torch.matrix_exp(B - B.T) @ q0).numpy()
        value = _rotation_score(candidate, z, D, dx, rank, other, cfg)
        if value['objective'] < best['objective']:
            best_Q, best = candidate.copy(), value
        if u == 0 or (u + 1) % 10 == 0 or u + 1 == steps:
            trace.append(dict(step=u + 1, **value))
    return best_Q, dict(**best, trace=trace, gradient_updates=steps)


def propose(audit, targets, cfg):
    """Bounded receiver/donor proposals; only the training audit is consulted.

    The receiver is the original capture owner, always pair[0]. Even when its
    detector owner differs, both candidate and control use the same coordinated
    receiver and common unedited heat calibration during fitting and decisions.
    """
    D = np.asarray(audit['atoms'], np.float64)
    rank = D.shape[1]
    oc = np.array([o['category'] for o in audit['objects']])
    rng = np.random.default_rng(cfg.get('proposal_seed', 71001))
    pool = []
    for t in targets:
        idx = np.flatnonzero(oc == t['category_index'])
        images = {audit['objects'][i]['image_id'] for i in idx}
        if len(images) < cfg['target_min_images']:
            continue
        dx = audit['dx'][idx]
        if float(np.sum(np.asarray(dx, float) ** 2)) <= 1e-12:
            continue
        receiver = int(t['contrast_group'])
        own = metrics.capture(dx, audit['dz'][idx, receiver] @ D[receiver])
        candidates = []
        for donor in range(D.shape[0]):
            if donor == receiver:
                continue
            cap = metrics.capture(dx, audit['dz'][idx, receiver] @ D[receiver] + audit['dz'][idx, donor] @ D[donor])
            candidates.append((cap - own, donor, cap))
        for gain, donor, cap in sorted(candidates, reverse=True)[:cfg.get('proposal_group_pool', 4)]:
            if gain < cfg.get('proposal_min_pair_gain', 0.):
                continue
            pool.append(dict(category_index=int(t['category_index']), category=t['category'],
                             receiver=receiver, donor=int(donor), pair=[receiver, int(donor)],
                             pair_capture=float(cap), original_receiver_capture=float(own), training_pair_gain=float(gain)))
    pool.sort(key=lambda p: (-p['training_pair_gain'], p['category_index'], p['receiver'], p['donor']))
    pair_limit = cfg.get('proposal_pairs', 2)
    pool = pool[:pair_limit]
    proposed = []
    for p in pool:
        began = time.monotonic()
        idx = np.flatnonzero(oc == p['category_index'])
        if len(idx) > cfg.get('proposal_object_limit', 512):
            idx = np.sort(rng.choice(idx, cfg.get('proposal_object_limit', 512), replace=False))
        pair = p['pair']
        z = audit['dz'][idx][:, pair].reshape(len(idx), 2 * rank).astype(np.float64)
        dx = audit['dx'][idx].astype(np.float64)
        dd = D[pair].reshape(2 * rank, -1)
        other = []
        for t in targets:
            if t['category_index'] == p['category_index'] or t['contrast_group'] not in pair:
                continue
            take = np.flatnonzero(oc == t['category_index'])
            if not len(take):
                continue
            if len(take) > cfg.get('proposal_collateral_object_limit', 64):
                take = np.sort(rng.choice(take, cfg.get('proposal_collateral_object_limit', 64), replace=False))
            vv = audit['dz'][take][:, pair].reshape(len(take), 2 * rank).astype(float)
            xx = audit['dx'][take].astype(float)
            owner = pair.index(t['contrast_group'])
            captured = metrics.capture(xx, vv[:, owner * rank:(owner + 1) * rank] @ dd[owner * rank:(owner + 1) * rank])
            if captured is None:
                continue
            other.append(dict(z=vv, dx=xx, local_owner=owner, capture=captured, category=t['category']))
        # Decoded target alignment supplies a deterministic initial subspace;
        # it does not use coefficient energy as a stand-in for object capture.
        alignment = z.T @ dx @ dd.T
        _, eigvec = np.linalg.eigh((alignment + alignment.T) / 2)
        aligned = _proper(eigvec[:, ::-1].T)
        rotations = []
        for name, seed in [('identity', np.eye(2 * rank)), ('decoded_aligned_pca', aligned)]:
            Q, diagnostics = _optimise_rotation(z, dd, dx, rank, other, seed, cfg)
            rotations.append((diagnostics['objective'], name, Q, diagnostics))
        _, source, Q, diagnostics = min(rotations, key=lambda item: item[0])
        # Also test the best proper, signed one-coordinate exchange. It has the
        # same decoder-aware objective and admits a half-strength homotopy.
        exchanges = []
        for i, j in itertools.product(range(rank), range(rank, 2 * rank)):
            exchange = np.eye(2 * rank)
            exchange[i, i] = exchange[j, j] = 0
            exchange[i, j], exchange[j, i] = 1., -1.
            value = _rotation_score(exchange, z, dd, dx, rank, other, cfg)
            exchanges.append((value['objective'], i, j, exchange, value))
        _, i, j, exchange, exchange_diagnostics = min(exchanges, key=lambda item: item[0])
        families = [('decoded_contrast_rotation', Q, dict(diagnostics, initialization=source,
                    initialization_comparisons=[dict(source=s, **a) for _, s, _, a in rotations])),
                    ('decoded_direction_exchange', exchange, dict(exchange_diagnostics, exchanged_local_coordinates=[i, j]))]
        # One family per pair keeps the default two-pair/two-strength diagnostic
        # at four fits. Both families are optimized/scored on training only.
        families = [min(families, key=lambda item: item[2]['objective'])]
        _, decoder_singular, decoder_axes = np.linalg.svd(dd, full_matrices=False)
        span_rank = int(np.sum(decoder_singular > max(decoder_singular[0] * 1e-8, 1e-12))) if len(decoder_singular) else 0
        B = decoder_axes[:span_rank].T
        singular = np.linalg.svd(dx @ B, compute_uv=False)
        energy = max(float(np.sum(dx ** 2)), 1e-12)
        capacity = dict(rank=rank, pair_span_rank=span_rank, pair_span_capture_upper_bound=float(np.sum(singular ** 2) / energy),
            rank_limited_capture_upper_bound=float(np.sum(singular[:rank] ** 2) / energy),
            interpretation='Training-only best linear rank-r reencoding inside the pair span; optimistic capacity diagnostic, not a native gated achievable score.')
        for kind, full, details in families:
            for strength in cfg.get('proposal_strengths', cfg.get('edit_strengths', [.5, 1.])):
                actual = orthogonal_homotopy(full, float(strength))
                value = _rotation_score(actual, z, dd, dx, rank, other, cfg)
                baseline = _rotation_score(np.eye(2 * rank), z, dd, dx, rank, other, cfg)
                proposed.append(dict(**p, kind=kind, Q=actual, strength=float(strength), Q_is_strength_applied=True,
                    training_surrogate=dict(**value, original_target_capture=baseline['target_capture'],
                        target_capture_gain=value['target_capture'] - baseline['target_capture'],
                        optimization=details, objects=len(idx), object_ids_sha=array_sha(idx),
                        collateral_categories=[o['category'] for o in other],
                        capacity=capacity, proposal_seconds=time.monotonic() - began,
                        total_rotation_gradient_updates=sum(a['gradient_updates'] for _, _, _, a in rotations),
                        gate='UNGATED coordinate-change surrogate using fixed observed native codes; not an acceptance score'),
                    training_rank_score=float(baseline['objective'] - value['objective']),
                    ownership_rule='Original contrast owner is receiving group. Assignments are frozen before selection/confirmation.'))
    proposed.sort(key=lambda p: (-p['training_rank_score'], p['category_index'], p['kind'], -p['strength']))
    return proposed[:cfg.get('candidates_per_round', 8)]


def targets_for_proposal(targets, proposal, audit):
    """One train-only assignment transaction, identically used for both arms."""
    out = copy.deepcopy(targets)
    if proposal is None:
        return out
    c, receiver = int(proposal['category_index']), int(proposal['receiver'])
    if proposal['pair'][0] != receiver or receiver == proposal['pair'][1]:
        raise ValueError('Receiver must be first in a distinct receiver/donor pair')
    matches = [t for t in out if t['category_index'] == c]
    if len(matches) != 1 or matches[0]['contrast_group'] != receiver:
        raise ValueError('Receiver must be the original training contrast owner')
    t = matches[0]
    h = np.asarray(audit['heat'][:, receiver], float)
    t.update(contrast_group=receiver, detector_group=receiver,
             heat_mean=float(h.mean()), heat_scale=max(float(h.std()), 1e-3),
             transaction_receiver=receiver, transaction_original_detector=int(t['detector_group']),
             calibration_source='Common unedited training audit, shared by candidate and control')
    return out


def score(weights, method, base, pack, targets, cfg):
    m = core.new_model(method, base, weights, cfg.get('device', 'cpu'))
    value = _assessment(m, pack, targets, cfg, cfg.get('device', 'cpu'))
    del m
    return value


def _assessment(model, pack, targets, cfg, device):
    """Aligned receiver metrics plus the original detector role, without remapping."""
    value = core.validation(model, pack, targets, cfg, device, with_arrays=True)
    audit = value.pop('_audit')
    y, valid = metrics.binary_labels(audit, cfg)
    rows = _rows(value)
    for t in targets:
        if 'transaction_original_detector' not in t:
            continue
        c, group = t['category_index'], t['transaction_original_detector']
        p, n = metrics.image_counts(y[:, c], valid[:, c], audit['image_index'])
        rows[c].update(original_detector_group=group,
                       original_detector_ap=metrics.ap(audit['heat'][:, group], y[:, c], valid[:, c])
                       if min(p, n) >= cfg['decision_min_images'] else None)
    return value


def _detector_role_guard(candidate, reference, cfg):
    """A detector may migrate only if the receiver preserves its actual quality."""
    old = reference.get('original_detector_ap')
    if 'original_detector_group' not in reference:
        return None
    now, receiver = candidate.get('original_detector_ap'), candidate.get('ap')
    if not all(_number(v) for v in (old, now, receiver)):
        return 'original_detector_role_insufficient_evidence'
    margin = cfg.get('category_preservation_tolerance', .03)
    if now < old - margin and receiver < old - margin:
        return 'target_detection_role_lost'
    return None


def _rows(value):
    return {r['category_index']: r for r in value['categories']}


def compare(candidate, control, initial, cfg, base, method, target_category=None):
    """Target gain plus explicit reconstruction/activity/other-object guards."""
    if target_category is None:
        raise ValueError('A coordinated edit needs an explicit target category')
    cc, oo, ii = _rows(candidate), _rows(control), _rows(initial)
    target = cc.get(target_category)
    original = oo.get(target_category)
    reasons = []
    min_images = cfg.get('decision_min_images', 10)
    insufficient = []
    cohort = set(oo) | set(ii)
    for c in sorted(cohort):
        for label, values in [('candidate', cc), ('control', oo), ('initial', ii)]:
            r = values.get(c, {})
            if not all(_number(r.get(f)) for f in ('capture', 'ap')):
                insufficient.append(dict(category_index=c, arm=label, reason='missing_capture_or_detection'))
            elif min(r.get('positive_images', min_images), r.get('negative_images', min_images)) < min_images:
                insufficient.append(dict(category_index=c, arm=label, reason='too_few_independent_images'))
    if insufficient:
        reasons.append('insufficient_annotation_evidence')
    gain = None if target is None or original is None or not _number(target.get('capture')) or not _number(original.get('capture')) else target['capture'] - original['capture']
    ap_gain = None if target is None or original is None or not _number(target.get('ap')) or not _number(original.get('ap')) else target['ap'] - original['ap']
    if gain is None or gain < _gain_threshold(cfg):
        reasons.append('target_capture_gain_too_small')
    if ap_gain is None or ap_gain < -_ap_margin(cfg):
        reasons.append('target_detection_declined')
    starting = ii.get(target_category, {})
    if target is not None:
        for field, tolerance in [('capture', cfg.get('macro_tradeoff_tolerance', .005)), ('ap', _ap_margin(cfg))]:
            if _number(target.get(field)) and _number(starting.get(field)) and target[field] < starting[field] - tolerance:
                reasons.append('target_' + field + '_declined_from_unedited')
    cm, m, im = control['reconstruction']['mse'], candidate['reconstruction']['mse'], initial['reconstruction']['mse']
    if not all(_number(v) and v >= 0 for v in (cm, m, im)):
        reasons.append('invalid_reconstruction_measurement')
    else:
        if m > cm * (1 + cfg.get('accept_reconstruction_tolerance', .01)):
            reasons.append('worse_reconstruction_than_matched_control')
        if m > im * (1 + cfg.get('global_reconstruction_tolerance', .02)):
            reasons.append('worse_reconstruction_than_unedited_start')
    activity = candidate['reconstruction'].get('active_groups')
    if method == 'group_lasso' and (not _number(activity) or abs(activity - base['active_groups']) > base['sparsity_tolerance'] * base['active_groups']):
        reasons.append('sparsity_mismatch')
    macro = []
    for field in ('macro_capture', 'macro_ap'):
        if not _number(candidate.get(field)) or not _number(control.get(field)):
            reasons.append('missing_' + field)
        elif candidate[field] < control[field] - cfg.get('macro_tradeoff_tolerance', .005) or (_number(initial.get(field)) and candidate[field] < initial[field] - cfg.get('macro_tradeoff_tolerance', .005)):
            reasons.append(field + '_declined')
    off = []
    for c in sorted(cohort - {target_category}):
        for field in ('capture', 'ap'):
            a, b, start = cc.get(c, {}).get(field), oo.get(c, {}).get(field), ii.get(c, {}).get(field)
            if not all(_number(v) for v in (a, b, start)):
                continue
            tolerance = cfg.get('category_preservation_tolerance', .03)
            # Neither a degraded continued-training control nor a fresh native
            # category remapping can lower the preservation bar unnoticed.
            if a < b - tolerance or a < start - tolerance:
                off.append(dict(category_index=c, category=cc[c].get('category'), metric=field,
                                candidate=a, control=b, unedited=start, delta=a - b,
                                delta_from_unedited=a - start))
    if off:
        reasons.append('other_object_categories_degraded')
    if target and original:
        for field in ('contrast_group', 'detector_group'):
            if target.get(field) != original.get(field):
                reasons.append('transaction_assignment_mismatch')
        for ref in (original, starting):
            problem = _detector_role_guard(target, ref, cfg)
            if problem:
                reasons.append(problem)
    thresholds = dict(target_capture_gain=_gain_threshold(cfg),
        target_ap_tolerance=_ap_margin(cfg),
        accept_reconstruction_tolerance=cfg.get('accept_reconstruction_tolerance', .01),
        global_reconstruction_tolerance=cfg.get('global_reconstruction_tolerance', .02),
        macro_tradeoff_tolerance=cfg.get('macro_tradeoff_tolerance', .005),
        category_preservation_tolerance=cfg.get('category_preservation_tolerance', .03))
    return dict(accepted=not reasons, reasons=sorted(set(reasons)), target_category=target_category,
                target_capture_gain=gain, target_ap_gain=ap_gain, composite_gain=None if gain is None or ap_gain is None else gain + cfg.get('score_ap_weight', .2) * ap_gain,
                reconstruction_relative_change=None if not _number(cm) or cm <= 0 else m / cm - 1,
                reconstruction_relative_to_unedited=None if not _number(im) or im <= 0 else m / im - 1,
                off_target_losses=off, insufficient_evidence=insufficient, thresholds=thresholds,
                candidate=candidate, control=control, initial=initial)


def preservation_only(candidate, initial, cfg, base, method):
    """Unedited fallback guard: every category is protected, with no exemptions.

    This does not test repair improvement. It only checks whether continued
    training is safe to retain instead of the frozen common start.
    """
    a, b = _rows(candidate), _rows(initial)
    cohort = set(b) | set(cfg.get('category_indices', []))
    reasons, checks, missing = [], [], []
    minimum = cfg.get('decision_min_images', 10)
    for c in sorted(cohort):
        current, ref = a.get(c, {}), b.get(c, {})
        for role, row in [('candidate', current), ('initial', ref)]:
            if not all(_number(row.get(f)) for f in ('capture', 'ap')):
                missing.append(dict(category_index=c, role=role, reason='missing_capture_or_detection'))
            elif min(row.get('positive_images', minimum), row.get('negative_images', minimum)) < minimum:
                missing.append(dict(category_index=c, role=role, reason='too_few_independent_images'))
        for field in ('capture', 'ap'):
            av, bv = current.get(field), ref.get(field)
            if not _number(av) or not _number(bv):
                continue
            margin = cfg.get('category_preservation_tolerance', .03)
            passed = av >= bv - margin
            checks.append(dict(category_index=c, category=ref.get('category'), metric=field,
                               candidate=av, initial=bv, delta=av - bv,
                               tolerance=margin, passed=bool(passed)))
            if not passed:
                reasons.append('object_category_' + field + '_declined')
        for field in ('contrast_group', 'detector_group'):
            if current.get(field) != ref.get(field):
                reasons.append('frozen_assignment_changed')
        problem = _detector_role_guard(current, ref, cfg)
        if problem:
            reasons.append(problem)
    if missing:
        reasons.append('insufficient_annotation_evidence')
    mse, ref_mse = candidate['reconstruction']['mse'], initial['reconstruction']['mse']
    if not all(_number(v) and v >= 0 for v in (mse, ref_mse)):
        reasons.append('invalid_reconstruction_measurement')
    elif mse > ref_mse * (1 + cfg.get('global_reconstruction_tolerance', .02)):
        reasons.append('worse_reconstruction_than_unedited_start')
    activity = candidate['reconstruction'].get('active_groups')
    if method == 'group_lasso' and (not _number(activity) or abs(activity - base['active_groups']) > base['sparsity_tolerance'] * base['active_groups']):
        reasons.append('sparsity_mismatch')
    for field in ('macro_capture', 'macro_ap'):
        av, bv = candidate.get(field), initial.get(field)
        if not _number(av) or not _number(bv):
            reasons.append('missing_' + field)
        elif av < bv - cfg.get('macro_tradeoff_tolerance', .005):
            reasons.append(field + '_declined')
    return dict(accepted=not reasons, passed=not reasons, reasons=sorted(set(reasons)),
                category_checks=checks, insufficient_evidence=missing,
                reconstruction_relative_to_unedited=None if not _number(ref_mse) or ref_mse <= 0 else mse / ref_mse - 1,
                rule='Preservation only: every frozen category is checked; no target/gain exemption.')


def _supervised(model, reference, x, masks, targets, head, cfg, proposal, update):
    """Ratio-of-sums capture, native gated detection and contextual anchoring."""
    ni, npats, d = x.shape
    z = model.encode(x.reshape(-1, d)).reshape(ni, npats, model.n_groups, model.group_size)
    D = model.decoder_atoms().reshape(model.n_groups, model.group_size, model.d)
    with torch.no_grad():
        rz = reference.encode(x.reshape(-1, d)).reshape_as(z)
        rD = reference.decoder_atoms().reshape_as(D)
    zero = z.sum() * 0
    w, ix, cats = core.object_weights(masks, [t['category_index'] for t in targets], cfg)
    capture = []
    target_capture = []
    preservation = []
    anchor_records = []
    stable_scales = cfg.get("stable_anchor_scales")
    def anchor(current, saved, role, denominator):
        if stable_scales is None:
            return (current-saved).square().sum() / saved.square().sum().clamp_min(cfg.get("anchor_energy_floor", 1e-6))
        from .stable import penalty
        return penalty(current, saved, denominator, cfg, anchor_records, role)
    if len(ix):
        ww = torch.as_tensor(w, device=x.device)
        images = torch.as_tensor(ix, device=x.device)
        dx = torch.einsum('mp,mpd->md', ww, x[images])
        dz = torch.einsum('mp,mpgr->mgr', ww, z[images])
        rdz = torch.einsum('mp,mpgr->mgr', ww, rz[images])
        for t in targets:
            take = torch.as_tensor(cats == t['category_index'], device=x.device)
            if not bool(take.any()):
                continue
            g = t['contrast_group']
            pred = dz[take, g] @ D[g]
            energy = dx[take].square().sum().clamp_min(1e-6)
            error = (dx[take] - pred).square().sum() / energy
            if proposal is not None and t["category_index"] == proposal["category_index"]:
                target_capture.append(error)
            else:
                capture.append(error)
            if proposal is not None and t['category_index'] == proposal['category_index']:
                continue
            # Other object roles retain their training-chosen group, including
            # roles in the edited pair. We protect those object contrasts rather
            # than freezing every response of both groups on target patches.
            ref = rdz[take, g] @ rD[g]
            den = stable_scales["category"][str(t["category_index"])]["denominator"] if stable_scales else 1.
            preservation.append(anchor(pred, ref, "category_"+str(t["category_index"]), den))
    frac = torch.as_tensor(np.stack([m['fraction'] for m in masks]).astype(np.float32) / 255, device=x.device)
    crowd = torch.as_tensor(np.stack([m['crowd'] for m in masks]), device=x.device)
    detection = []
    target_detection = []
    for j, t in enumerate(targets):
        f = frac[:, :, t['category_index']]
        pos, neg = f >= cfg['positive_fraction'], f <= cfg['negative_fraction']
        valid = (pos | neg) & (crowd <= cfg['negative_fraction'])
        pos, neg = pos & valid, neg & valid
        heat = (z[:, :, t['detector_group']].norm(dim=-1) - t['heat_mean']) / t['heat_scale']
        logits = F.softplus(head[j, 0]) * heat + head[j, 1]
        terms = []
        if bool(pos.any()):
            terms.append(F.softplus(-logits[pos]).mean())
        if bool(neg.any()):
            terms.append(F.softplus(logits[neg]).mean())
        if terms:
            value = torch.stack(terms).mean()
            if proposal is not None and t["category_index"] == proposal["category_index"]:
                target_detection.append(value)
            else:
                detection.append(value)
    # Protect strong untouched groups on a deterministic round-robin subset of
    # patches. No full all-group contribution cache is constructed.
    pair = set(proposal['pair']) if proposal is not None else set()
    groups = {int(t['contrast_group']) for t in targets
                     if t['contrast_group'] not in pair and t.get('training_capture', 0.) is not None
                     and t.get('training_capture', 0.) >= cfg.get('protect_capture_threshold', .5)}
    detector_groups = {int(t.get('transaction_original_detector', t['detector_group'])) for t in targets
                       if t.get('training_ap', 0.) is not None and t.get('training_ap', 0.) >= cfg.get('protect_ap_threshold', .8)
                       and t.get('transaction_original_detector', t['detector_group']) not in pair}
    groups = sorted(groups | detector_groups)
    count = min(len(groups), cfg.get('preserve_groups_per_update', 4))
    use = [groups[(update * count + j) % len(groups)] for j in range(count)] if count else []
    stride = max(1, (ni * npats) // cfg.get('preserve_patch_limit', 128))
    flat_z, flat_rz = z.reshape(-1, model.n_groups, model.group_size)[::stride], rz.reshape(-1, model.n_groups, model.group_size)[::stride]
    for g in use:
        current = flat_z[:, g] @ D[g]
        saved = flat_rz[:, g] @ rD[g]
        den = stable_scales["contribution"][g] if stable_scales else 1.
        preservation.append(anchor(current, saved, "group_"+str(g), den))
        if g in detector_groups:
            heat, old_heat = flat_z[:, g].norm(dim=-1), flat_rz[:, g].norm(dim=-1)
            den = stable_scales["heat"][g] if stable_scales else 1.
            preservation.append(anchor(heat, old_heat, "heat_"+str(g), den))
    context = zero
    context_patches = 0
    if proposal is not None:
        pair = proposal['pair']
        known = crowd <= cfg['negative_fraction']
        not_target = (frac[:, :, proposal['category_index']] <= cfg['negative_fraction']) & known
        context_patches = int(not_target.sum())
        if bool(not_target.any()):
            current = sum(z[:, :, g] @ D[g] for g in pair)[not_target]
            saved = sum(rz[:, :, g] @ rD[g] for g in pair)[not_target]
            den = sum(stable_scales["contribution"][g] for g in pair) if stable_scales else 1.
            context = anchor(current, saved, "pair_context", den)
    if cfg.get("target_capture_mix") is not None:
        mix = cfg["target_capture_mix"]
        capture_loss = mix * (torch.stack(target_capture).mean() if target_capture else zero) + (1-mix) * (torch.stack(capture).mean() if capture else zero)
    else:
        all_capture = capture + target_capture
        capture_loss = torch.stack(all_capture).mean() if all_capture else zero
    if cfg.get("target_detection_mix") is not None:
        mix = cfg["target_detection_mix"]
        detection_loss = mix * (torch.stack(target_detection).mean() if target_detection else zero) + (1-mix) * (torch.stack(detection).mean() if detection else zero)
    else:
        all_detection = detection + target_detection
        detection_loss = torch.stack(all_detection).mean() if all_detection else zero
    losses = dict(capture=capture_loss,
                  detection=detection_loss,
                  preservation=torch.stack(preservation).mean() if preservation else zero,
                  pair_context=context)
    support = z.norm(dim=-1) > 1e-6
    previous = rz.norm(dim=-1) > 1e-6
    union = (support | previous).sum().clamp_min(1)
    diagnostics = dict(objects_in_batch=int(len(ix)), represented_categories_in_batch=len(capture)+len(target_capture),
                       support_jaccard=float(((support & previous).sum() / union).detach()),
                       support_churn=float((support != previous).float().mean().detach()),
                       active_groups=float(support.sum(-1).float().mean().detach()),
                       untouched_group_replay_count=len(use), pair_context_patches=context_patches,
                       anchor_diagnostics=anchor_records, target_capture_loss=float(torch.stack(target_capture).mean().detach()) if target_capture else None)
    if proposal is not None:
        diagnostics.update(receiver_active_fraction=float(support[:, :, proposal['receiver']].float().mean().detach()),
                           donor_active_fraction=float(support[:, :, proposal['donor']].float().mean().detach()))
    return losses, diagnostics


def _gradients(model, terms):
    params = [p for p in model.parameters() if p.requires_grad]
    vectors, norms = {}, {}
    for name, term in terms.items():
        grads = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
        v = torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)])
        vectors[name], norms[name] = v, float(v.norm().detach())
    cosine = {}
    for a, b in itertools.combinations(vectors, 2):
        den = vectors[a].norm() * vectors[b].norm()
        cosine[a + '_vs_' + b] = None if float(den) <= 1e-12 else float((vectors[a] @ vectors[b] / den).detach())
    return dict(weighted_loss_gradient_norms=norms, gradient_cosines=cosine,
                interpretation='Negative cosine is local gradient tension; it is not evidence of a causal failure by itself.')


def _eligible(value, reference, cfg, base, method, category):
    """Checkpoint guards use the UNEDITED start, never the damaged edit."""
    reasons = []
    if value['reconstruction']['mse'] > reference['reconstruction']['mse'] * (1 + cfg.get('global_reconstruction_tolerance', .02)):
        reasons.append('reconstruction_above_unedited_start')
    if method == 'group_lasso' and abs(value['reconstruction']['active_groups'] - base['active_groups']) > base['sparsity_tolerance'] * base['active_groups']:
        reasons.append('sparsity_mismatch')
    a, b = _rows(value), _rows(reference)
    for c in b:
        if c not in a or any(not _number(a[c].get(f)) or not _number(b[c].get(f)) for f in ('capture', 'ap')):
            reasons.append('insufficient_annotation_evidence')
            continue
        if c != category:
            for field in ('capture', 'ap'):
                if a[c][field] < b[c][field] - cfg.get('category_preservation_tolerance', .03):
                    reasons.append('other_object_categories_degraded')
    if category is not None and category in a and category in b:
        role_problem = _detector_role_guard(a[category], b[category], cfg)
        if role_problem:
            reasons.append(role_problem)
        if _number(a[category].get('capture')) and _number(b[category].get('capture')) and a[category]['capture'] < b[category]['capture'] - cfg.get('macro_tradeoff_tolerance', .005):
            reasons.append('target_capture_declined_from_unedited')
        if _number(a[category].get('ap')) and _number(b[category].get('ap')) and a[category]['ap'] < b[category]['ap'] - _ap_margin(cfg):
            reasons.append('target_detection_declined')
    for field in ('macro_capture', 'macro_ap'):
        if not _number(value.get(field)) or not _number(reference.get(field)):
            reasons.append('insufficient_annotation_evidence')
        elif value[field] < reference[field] - cfg.get('macro_tradeoff_tolerance', .005):
            reasons.append(field + '_declined')
    return not reasons, sorted(set(reasons))


def fit(store, key, method, base, initial, train_pack, select_pack, targets, cfg, seed, proposal=None):
    """Fresh candidate/control optimizers, local+Drive exact-state resume.

    Optional cfg['_max_updates'] is for interruption/resume tests; changing it
    does not alter the declared training horizon or checkpoint identity.
    """
    device = cfg.get('device', 'cpu')
    if set(train_pack['ids']) & set(select_pack['ids']):
        raise ValueError('Training/selection image overlap')
    target_updates = cfg.get('updates', cfg.get('refit_updates', 500))
    category = proposal['category_index'] if proposal is not None else cfg.get('transaction_category')
    pair_context_proposal = proposal if proposal is not None else cfg.get('control_context_proposal')
    torch.manual_seed(seed)
    if device.startswith('cuda'):
        torch.cuda.manual_seed_all(seed)
    source_sha = training.weight_sha(initial)
    Q = None
    if proposal is not None:
        Q = proposal['Q'] if proposal.get('Q_is_strength_applied') else orthogonal_homotopy(proposal['Q'], proposal.get('strength', 1.))
        # An identity edit must not re-QR a Grassmannian frame and introduce
        # roundoff before comparing identical controls.
        edited = initial if np.array_equal(Q, np.eye(Q.shape[0])) else core.transform_weights(method, initial, proposal['pair'], Q, base['block_rank'])
    else:
        edited = initial
    reference = core.new_model(method, base, initial, device).eval()
    for p in reference.parameters():
        p.requires_grad_(False)
    model = core.new_model(method, base, edited, device)
    head = torch.nn.Parameter(torch.zeros(len(targets), 2, device=device))
    local_masks = None
    if cfg.get('local_pair_recovery'):
        from .stable import pair_masks
        if pair_context_proposal is None: raise ValueError('Local recovery requires an explicit pair')
        local_masks = pair_masks(model, pair_context_proposal['pair'])
    opt = torch.optim.Adam(list(model.parameters()) + [head], lr=cfg['lr'])
    rng = np.random.default_rng(seed)
    identity_cfg = {k: v for k, v in cfg.items() if k != '_max_updates'}
    fit_id = identity(dict(method=method, source_sha=source_sha, edited_sha=training.weight_sha(edited),
                           targets=targets, config=identity_cfg, proposal=proposal, seed=seed,
                           train_sha=_array_identity(train_pack['x']), select_sha=_array_identity(select_pack['x'])))
    state = store.load(key)
    if state is None:
        store.event('coordinated_evaluation_start', key=key, update=0, role='unedited_reference')
        ref = _assessment(reference, select_pack, targets, cfg, device)
        immediate = _assessment(model, select_pack, targets, cfg, device)
        eligible, reasons = _eligible(immediate, ref, cfg, base, method, category)
        row = _rows(immediate).get(category, {})
        objective = row.get('capture') if category is not None else immediate.get('macro_capture')
        state = dict(fit_id=fit_id, initial_sha=source_sha, edited_initial_sha=training.weight_sha(edited),
                     updates=0, target=target_updates, best=training.weights(model),
                     best_score=objective if eligible else None, best_update=0 if eligible else None,
                     best_valid=bool(eligible), best_validation=immediate if eligible else None,
                     reference_validation=ref, seconds=0., proposal=proposal, targets=copy.deepcopy(targets),
                     gradient_history=[],
                     history=[dict(update=0, validation=immediate, checkpoint_eligible=eligible,
                         eligibility_reasons=reasons, immediate_reconstruction_relative_spike=immediate['reconstruction']['mse'] / max(ref['reconstruction']['mse'], 1e-12) - 1)],
                     weights=training.weights(model), head=head.detach().cpu().clone())
    else:
        if state['fit_id'] != fit_id:
            raise ValueError('Fit identity changed; preserve the old run and use a new identity')
        model.load_state_dict(state['weights'])
        head.data.copy_(state['head'].to(device))
        opt.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['sample_rng']
        torch.set_rng_state(state['torch_rng'])
        if device.startswith('cuda') and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
    replay_pool = None
    if cfg.get('target_replay_fraction', 0):
        replay_pool = []
        for i, mask in enumerate(train_pack['masks']):
            _, present, _ = core.object_weights([mask], [category], cfg)
            if len(present): replay_pool.append(i)
        if not replay_pool: raise ValueError('No eligible target objects for training replay')
        replay_pool = np.array(replay_pool)
        store.event('training_target_replay',key=key,category=category,eligible_images=len(replay_pool),pool_sha=array_sha(replay_pool),fraction=cfg['target_replay_fraction'])
    limit = min(target_updates, cfg.get('_max_updates', target_updates))
    print(key, 'resume', state['updates'], '/', target_updates, flush=True)
    store.event('coordinated_fit_start', key=key, update=state['updates'], target=target_updates,
                source_sha=source_sha, edited_initial_sha=state['edited_initial_sha'],
                category=category, receiver=None if proposal is None else proposal['receiver'])

    def save(drive=True):
        state.update(weights=training.weights(model), head=head.detach().cpu().clone(),
                     optimizer=copy.deepcopy(opt.state_dict()), sample_rng=copy.deepcopy(rng.bit_generator.state),
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)
        store.save(key, state, drive)

    try:
        while state['updates'] < limit:
            ix = rng.integers(len(train_pack['x']), size=cfg['images_per_update'])
            if replay_pool is not None:
                count = max(1, round(len(ix) * cfg['target_replay_fraction']))
                ix[:count] = rng.choice(replay_pool, size=count, replace=True)
                rng.shuffle(ix)
            x = torch.as_tensor(train_pack['x'][ix], dtype=torch.float32, device=device)
            model.train()
            started = time.monotonic()
            xx = x.reshape(-1, model.d)
            noisy = xx + cfg.get('snr', 0.) * xx.std() * torch.randn_like(xx) if cfg.get('snr', 0.) else xx
            native, info = model.loss(noisy, target=xx)
            losses, batch = _supervised(model, reference, x, [train_pack['masks'][i] for i in ix], targets, head, cfg, pair_context_proposal, state['updates'])
            batch['training_batch_indices_sha'] = array_sha(ix)
            batch['target_replay_images'] = count if replay_pool is not None else 0
            terms = dict(native=native, capture=cfg['lambda_capture'] * losses['capture'],
                         detection=cfg['lambda_detection'] * losses['detection'],
                         preservation=cfg.get('lambda_preservation', .02) * losses['preservation'],
                         pair_context=cfg.get('lambda_pair_context', .02) * losses['pair_context'])
            total = sum(terms.values())
            if not torch.isfinite(total):
                raise FloatingPointError('Nonfinite coordinated adaptation loss')
            diagnostics = _gradients(model, terms) if (state['updates'] + 1) % cfg.get('diagnostic_grad_every', 50) == 0 else None
            opt.zero_grad()
            total.backward()
            if local_masks is not None:
                from .stable import mask_gradients
                mask_gradients(model, local_masks)
            grad = float(torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [head], cfg.get('grad_clip', 10.)))
            encoder_grad = float(sum(p.grad.detach().square().sum() for name, p in model.named_parameters() if p.grad is not None and ('enc' in name or name == 'gamma')).sqrt())
            decoder_grad = float(sum(p.grad.detach().square().sum() for name, p in model.named_parameters() if p.grad is not None and name in ('W_dec', 'B_raw')).sqrt())
            opt.step()
            model.normalize_decoder()
            if local_masks is not None:
                from .stable import restore_outside_pair, outside_pair_drift
                restore_outside_pair(model, local_masks, edited)
                batch['outside_pair_parameter_drift'] = outside_pair_drift(model, local_masks, edited)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            state['seconds'] += time.monotonic() - started
            state['updates'] += 1
            u = state['updates']
            if diagnostics is not None:
                state.setdefault('gradient_history', []).append(dict(update=u, **diagnostics))
            if u % cfg.get('evaluate_every', 50) == 0 or u == target_updates:
                store.event('coordinated_evaluation_start', key=key, update=u, role='selection_only')
                value = _assessment(model, select_pack, targets, cfg, device)
                eligible, reasons = _eligible(value, state['reference_validation'], cfg, base, method, category)
                target = _rows(value).get(category, {})
                if cfg.get('retain_immediate_gain') and category is not None:
                    immediate_row = _rows(state['history'][0]['validation']).get(category, {})
                    for field in ('capture', 'ap'):
                        if _number(immediate_row.get(field)) and _number(target.get(field)) and target[field] < immediate_row[field] - cfg.get('immediate_retention_tolerance', .002):
                            reasons.append('immediate_'+field+'_gain_lost')
                    eligible = eligible and not reasons
                objective = target.get('capture') if category is not None else value.get('macro_capture')
                if eligible and _number(objective) and (state['best_score'] is None or objective > state['best_score']):
                    state.update(best=training.weights(model), best_score=float(objective), best_update=u,
                                 best_valid=True, best_validation=value)
                row = dict(update=u, validation=value, checkpoint_eligible=bool(eligible), eligibility_reasons=reasons,
                           total_loss=float(total.detach()), native_loss=float(native.detach()),
                           **{name + '_loss': float(loss.detach()) for name, loss in losses.items()},
                           gradient_norm=grad, encoder_gradient_norm=encoder_grad, decoder_gradient_norm=decoder_grad,
                           component_gradients=diagnostics, latest_component_gradients=state.get('gradient_history', [])[-1] if state.get('gradient_history') else None,
                           actual_gate_diagnostics=batch,
                           native_loss_info=info, seconds_per_update=state['seconds'] / u,
                           reconstruction_relative_to_unedited=value['reconstruction']['mse'] / max(state['reference_validation']['reconstruction']['mse'], 1e-12) - 1)
                state['history'].append(row)
                print(key, u, '/', target_updates, 'target capture', target.get('capture'), 'eligible', eligible, 'best', state['best_update'], flush=True)
                save(drive=(u % cfg.get('drive_save_every', 100) == 0 or u == target_updates))
                store.write('histories/' + key + '.json', state['history'])
                store.write('gradients/' + key + '.json', state.get('gradient_history', []))
                store.event('coordinated_checkpoint', key=key, update=u, target=target_updates,
                            eligible=eligible, reasons=reasons, target_capture=target.get('capture'),
                            target_ap=target.get('ap'), macro_capture=value['macro_capture'],
                            macro_ap=value['macro_ap'], mse=value['reconstruction']['mse'],
                            best_update=state['best_update'], seconds_per_update=row['seconds_per_update'])
            elif u % cfg.get('local_save_every', 20) == 0:
                save(False)
                store.event('coordinated_heartbeat', key=key, update=u, target=target_updates,
                            compute_seconds=state['seconds'], actual_gate_diagnostics=batch)
        save()
    except BaseException as exc:
        save(False)
        try:
            save()
            store.event('coordinated_fit_interrupted', key=key, update=state['updates'],
                        error_type=type(exc).__name__, error=str(exc))
        except BaseException:
            pass
        raise
    del model, reference, head, opt
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    return state
