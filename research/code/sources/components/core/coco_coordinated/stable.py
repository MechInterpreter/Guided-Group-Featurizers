"""Training-calibrated anchoring and explicitly restricted recovery.

Floors and robust residuals are exploratory controls, not a claim of optimality.
Every denominator is a fixed training statistic shared by edited/fixed arms.
"""
import numpy as np
import torch
import torch.nn.functional as F


def scales(audit, targets, cfg):
    atoms = np.asarray(audit['atoms'], np.float64)
    cats = np.array([o['category'] for o in audit['objects']])
    dx = np.asarray(audit['dx'], np.float64)
    heat = np.asarray(audit['heat'], np.float64)
    heat_energy = np.mean(heat ** 2, axis=0)
    # A unit coefficient's typical decoded energy; no giant all-patch cache.
    atom_energy = np.mean(np.sum(atoms ** 2, axis=1), axis=1) / atoms.shape[1]
    group_energy = heat_energy * atom_energy
    relative_floor = cfg.get('anchor_relative_floor', .01)
    absolute_floor = cfg.get('anchor_absolute_floor', 1e-6)
    group_floor = max(float(group_energy.mean()) * relative_floor, absolute_floor)
    heat_floor = max(float(heat_energy.mean()) * relative_floor, absolute_floor)
    category = {}
    for t in targets:
        c, g = t['category_index'], t['contrast_group']
        take = cats == c
        if not take.any():
            raise ValueError('Training scale lacks category '+str(c))
        reference = audit['dz'][take, g].astype(np.float64) @ atoms[g]
        observed = float(np.mean(dx[take] ** 2))
        energy = float(np.mean(reference ** 2))
        category[str(c)] = dict(reference_energy=energy, observation_energy=observed,
            denominator=max(energy, relative_floor * observed, absolute_floor),
            floor_active=energy < max(relative_floor * observed, absolute_floor),
            objects=int(take.sum()))
    result = dict(category=category,
        contribution=np.maximum(group_energy, group_floor).tolist(),
        heat=np.maximum(heat_energy, heat_floor).tolist(),
        group_floor=group_floor, heat_floor=heat_floor,
        calibration='Unedited training audit only; frozen for candidates and controls',
        contribution_scale_note='Heat energy times mean atom energy is a fixed scale surrogate, not exact decoded contribution energy.')
    if not all(np.isfinite(result[k]).all() and np.min(result[k]) > 0 for k in ('heat','contribution')):
        raise FloatingPointError('Invalid frozen anchor scales')
    return result


def penalty(current, reference, denominator, cfg, records, role):
    denominator = float(denominator)
    if not np.isfinite(denominator) or denominator <= 0:
        raise FloatingPointError('Invalid anchor denominator')
    normalized = (current - reference.detach()) / denominator ** .5
    beta = cfg.get('anchor_huber_beta', 1.)
    # Twice Huber equals squared error near zero and has bounded residual slope.
    value = 2 * F.smooth_l1_loss(normalized, torch.zeros_like(normalized), beta=beta)
    records.append(dict(role=role, denominator=denominator,
        batch_reference_energy=float(reference.detach().square().mean()),
        reference_active_fraction=float((reference.detach().abs() > 1e-6).float().mean()),
        normalized_residual_max=float(normalized.detach().abs().max()),
        robust_tail_fraction=float((normalized.detach().abs() > beta).float().mean()),
        loss=float(value.detach())))
    return value


def pair_masks(model, pair):
    """Parameter masks, including Group Lasso thresholds and tied QR frames."""
    out = {}
    r = model.group_size
    for name, p in model.named_parameters():
        mask = torch.zeros_like(p)
        rows = [g*r+j for g in pair for j in range(r)]
        if name == 'W_enc': mask[:, rows] = 1
        elif name in ('W_dec','b_enc'): mask[rows] = 1
        elif name in ('B_raw','gamma','raw_theta'): mask[pair] = 1
        else:
            raise ValueError('No local recovery mask for parameter '+name)
        out[name] = mask
    return out


def mask_gradients(model, masks):
    for name, p in model.named_parameters():
        if p.grad is not None: p.grad.mul_(masks[name])


@torch.no_grad()
def restore_outside_pair(model, masks, edited):
    # Normalizing every decoder block can move nominally frozen rows. Restore
    # them exactly, after normalization. A fresh optimizer has zero outside moments.
    for name, p in model.named_parameters():
        p.copy_(torch.where(masks[name].bool(), p, edited[name].to(p.device)))


def outside_pair_drift(model, masks, edited):
    with torch.no_grad():
        return max(float(((p-edited[name].to(p.device))*(1-masks[name])).abs().max())
                   for name,p in model.named_parameters())
