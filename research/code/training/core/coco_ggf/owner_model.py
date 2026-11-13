"""Supervised category ownership. Masks never enter encode/forward/routing.

An owner is a slot assigned to an annotated category, not a certificate that
all of that category's semantic information resides in that slot.
"""

import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from bsf import VanillaBSF
from coco_adaptation import core
from coco_baselines import metrics

MODES = ("continued_ggf", "owner_topk", "owner_router")


def distinct_owners(capture, ap):
    """Maximize train-only capture + .2 AP under one distinct slot/category.

    Exact assignment for this measured score matrix, not optimal semantics.
    """
    capture, ap = np.asarray(capture, float), np.asarray(ap, float)
    if (
        capture.shape != ap.shape
        or capture.ndim != 2
        or capture.shape[0] > capture.shape[1]
    ):
        raise ValueError("Need matching category-by-group matrices and enough groups")
    score = capture + 0.2 * ap
    if not np.isfinite(score).all():
        raise ValueError("Every cohort category/group needs finite training evidence")
    rows, cols = linear_sum_assignment(-score)
    order = np.argsort(rows)
    owners = cols[order]
    if len(set(owners.tolist())) != len(owners):
        raise RuntimeError("Owner assignment collision")
    return owners, score[np.arange(len(owners)), owners]


def category_capture_matrix(audit, cohort, chunk=64):
    """Actual one-block contrast capture, without refitting codes or decoders."""
    D = np.asarray(audit["atoms"], np.float64)
    gram = np.einsum("grd,gsd->grs", D, D)
    cats = np.array([o["category"] for o in audit["objects"]])
    result = np.zeros((len(cohort), len(D)), np.float64)
    for j, c in enumerate(cohort):
        ids = np.flatnonzero(cats == c)
        if not len(ids):
            raise ValueError("Missing training objects for category " + str(c))
        gain = np.zeros(len(D), np.float64)
        den = 0.0
        for lo in range(0, len(ids), chunk):
            ix = ids[lo : lo + chunk]
            dx = np.asarray(audit["dx"][ix], np.float64)
            dz = np.asarray(audit["dz"][ix], np.float64)
            projection = (dx @ D.reshape(-1, D.shape[-1]).T).reshape(
                len(ix), len(D), D.shape[1]
            )
            gain += 2 * np.einsum("mgr,mgr->g", dz, projection) - np.einsum(
                "mgr,grs,mgs->g", dz, gram, dz
            )
            den += float(np.square(dx).sum())
        if den <= 1e-12:
            raise ValueError("Zero contrast evidence for category " + str(c))
        result[j] = gain / den
    return result


def calibrated_targets(audit, categories, cohort, owners, capture_matrix):
    """Training-only owner readouts and logistic-head initialization."""
    y, valid = metrics.binary_labels(audit, categories["cfg"])
    cfg = categories["cfg"]
    rows = []
    head = []
    for j, (c, g) in enumerate(zip(cohort, owners)):
        h = audit["heat"][:, g].astype(float)
        if not y[valid[:, c], c].any() or y[valid[:, c], c].all():
            raise ValueError("Missing positive/negative training patches")
        threshold = metrics.f1_threshold(h, y[:, c], valid[:, c])
        mean, scale = float(h.mean()), max(float(h.std()), 1e-3)
        rows.append(
            dict(
                category_index=int(c),
                category=categories["names"][c],
                contrast_group=int(g),
                detector_group=int(g),
                heat_mean=mean,
                heat_scale=scale,
                training_capture=float(capture_matrix[j, g]),
                training_threshold=threshold,
            )
        )
        head.append([float(np.log(np.expm1(1.0))), -(threshold - mean) / scale])
    return rows, torch.tensor(head, dtype=torch.float32)


class OwnerGGF(VanillaBSF):
    """Native TopK or independent category gates plus a residual TopK bank.

    All modes retain identical dictionary shapes. Router heads have two learned
    parameters/category, just like the existing GGF detector training heads.
    With routed inference these parameters are deployed, so timing is charged.
    """

    def __init__(
        self, weights, groups, rank, active, targets, head, mode, router_targets=None
    ):
        d = weights["W_dec"].shape[1]
        super().__init__(d=d, n_groups=groups, group_size=rank, l0=active)
        if mode not in MODES or active < 1 or active > groups:
            raise ValueError("Invalid mode or active budget")
        self.load_state_dict(weights)
        self.mode = mode
        self.targets = copy.deepcopy(targets)
        owners = [t["contrast_group"] for t in targets]
        detectors = [t["detector_group"] for t in targets]
        if mode != "continued_ggf" and (
            len(set(owners)) != len(owners) or owners != detectors
        ):
            raise ValueError("Owner arms need distinct, shared capture/detection IDs")
        if any(g < 0 or g >= groups for g in owners + detectors):
            raise ValueError("Owner outside dictionary")
        if len(owners) >= groups and mode == "owner_router":
            raise ValueError("Keep a residual bank")
        self.register_buffer("owner_ids", torch.tensor(owners, dtype=torch.long))
        self.register_buffer("detector_ids", torch.tensor(detectors, dtype=torch.long))
        rt = router_targets if mode == "owner_router" else targets
        if rt is None or len(rt) != len(targets):
            raise ValueError("Missing training-only router calibration")
        self.register_buffer("heat_mean", torch.tensor([t["heat_mean"] for t in rt]))
        self.register_buffer("heat_scale", torch.tensor([t["heat_scale"] for t in rt]))
        self.head = nn.Parameter(head.detach().clone())
        if self.head.shape != (len(targets), 2):
            raise ValueError("Wrong head shape")
        residual = torch.ones(groups, dtype=torch.bool)
        residual[self.owner_ids.cpu()] = False
        self.register_buffer("residual_groups", residual)

    def raw_codes(self, x):
        return (x @ self.W_enc + self.b_enc).reshape(
            len(x), self.n_groups, self.group_size
        )

    def logits_from_heat(self, heat):
        h = (heat - self.heat_mean) / self.heat_scale
        return h * F.softplus(self.head[:, 0]) + self.head[:, 1]

    def routed(self, x, diagnostics=False):
        raw = self.raw_codes(x)
        logits = self.logits_from_heat(raw[:, self.owner_ids].norm(dim=-1))
        probabilities = logits.sigmoid()
        # Independent binary gates; there is no across-category softmax.
        requested = logits >= 0
        kk = min(self.l0, len(self.owner_ids))
        order = probabilities.topk(kk, dim=1).indices
        selected = torch.zeros_like(probabilities).scatter(1, order, 1.0)
        admitted = selected * requested.to(probabilities.dtype)
        # STE provides gate gradients, but FORWARD ALWAYS uses the real hard gate.
        owner_gate = (
            admitted + probabilities - probabilities.detach()
            if self.training
            else admitted
        )
        group_gate = raw.new_zeros((len(x), self.n_groups))
        group_gate[:, self.owner_ids] = owner_gate
        remaining = self.l0 - admitted.sum(1).to(torch.long)
        residual_score = raw.norm(dim=-1).masked_fill(
            ~self.residual_groups[None], -torch.inf
        )
        kr = min(self.l0, int(self.residual_groups.sum()))
        ri = residual_score.topk(kr, dim=1).indices
        rm = (torch.arange(kr, device=x.device)[None] < remaining[:, None]).to(
            raw.dtype
        )
        group_gate = group_gate.scatter(1, ri, rm)
        z = raw * group_gate[:, :, None]
        if not diagnostics:
            return z, logits
        return (
            z,
            logits,
            dict(
                requested_owner_count=requested.sum(1),
                admitted_owner_count=admitted.sum(1),
                truncated_owner_count=requested.sum(1) - admitted.sum(1),
                residual_count=rm.sum(1),
                owner_admitted=admitted,
            ),
        )

    def encode(self, x):
        if self.mode == "owner_router":
            return self.routed(x)[0]
        return super().encode(x)

    def detection_logits(self, x, z=None):
        if self.mode == "owner_router":
            raw = self.raw_codes(x)
            return self.logits_from_heat(raw[:, self.owner_ids].norm(dim=-1))
        z = self.encode(x) if z is None else z
        return self.logits_from_heat(z[:, self.detector_ids].norm(dim=-1))


def supervised_terms(model, x, masks, cfg, details=False):
    """Existing contrast target and balanced detection, using declared readouts.

    Object masks are used ONLY in losses. They never supply route decisions.
    """
    n, patches, d = x.shape
    flat = x.reshape(-1, d)
    if model.mode == "owner_router":
        z, logits = model.routed(flat)
    else:
        z = model.encode(flat)
        logits = model.detection_logits(flat, z)
    z = z.reshape(n, patches, model.n_groups, model.group_size)
    logits = logits.reshape(n, patches, len(model.targets))
    D = model.decoder_atoms().reshape(model.n_groups, model.group_size, d)
    weights, image, cats = core.object_weights(
        masks, [t["category_index"] for t in model.targets], cfg
    )
    cap = []
    det = []
    counts = {}
    if len(image):
        w = torch.as_tensor(weights, dtype=x.dtype, device=x.device)
        ix = torch.as_tensor(image, device=x.device)
        dx = torch.einsum("mp,mpd->md", w, x[ix])
        dz = torch.einsum("mp,mpgr->mgr", w, z[ix])
        for t in model.targets:
            take = torch.as_tensor(cats == t["category_index"], device=x.device)
            if not take.any():
                continue
            prediction = dz[take, t["contrast_group"]] @ D[t["contrast_group"]]
            cap.append(
                (
                    (dx[take] - prediction).square().sum(-1)
                    / dx[take].square().sum(-1).clamp_min(1e-6)
                ).mean()
            )
    fractions = torch.as_tensor(
        np.stack([m["fraction"] for m in masks]).astype(np.float32) / 255,
        device=x.device,
    )
    crowd = torch.as_tensor(np.stack([m["crowd"] for m in masks]), device=x.device)
    for j, t in enumerate(model.targets):
        f = fractions[:, :, t["category_index"]]
        valid = ((f >= cfg["positive_fraction"]) | (f <= cfg["negative_fraction"])) & (
            crowd <= cfg["negative_fraction"]
        )
        pos = (f >= cfg["positive_fraction"]) & valid
        neg = (f <= cfg["negative_fraction"]) & valid
        terms = []
        if pos.any():
            terms.append(F.softplus(-logits[:, :, j][pos]).mean())
        if neg.any():
            terms.append(F.softplus(logits[:, :, j][neg]).mean())
        if terms:
            det.append(torch.stack(terms).mean())
        if details:
            counts[str(t["category_index"])] = dict(
                positive=int(pos.sum()), negative=int(neg.sum())
            )
    zero = z.sum() * 0
    return (
        torch.stack(cap).mean() if cap else zero,
        torch.stack(det).mean() if det else zero,
        len(image),
        counts,
    )
