"""Frozen native concept interfaces. No training, new heads or BSF conversion."""

import math
import numpy as np
import torch


@torch.inference_mode()
def partial_nnls(x, blocks, tolerance=1e-5, iterations=2000):
    """Independent per-concept NNLS, with original-space float64 KKT checks.

    These independent projections are not summed or treated as a joint additive
    reconstruction. Only the requested dictionaries are evaluated.
    """
    if tolerance != 1e-5:
        raise ValueError("Keep the completed SLiCS tolerance")
    B = blocks
    g, b, d = B.shape
    if x.ndim != 2 or x.shape[1] != d:
        raise ValueError("SLiCS input dimensions")
    gram = B @ B.transpose(1, 2)
    cross = (x @ B.reshape(-1, d).T).reshape(len(x), g, b)
    L = torch.linalg.eigvalsh(gram).amax(-1).clamp_min(1e-8)[None, :, None]
    z = torch.zeros_like(cross)
    y = z.clone()
    t = 1.0
    for it in range(iterations):
        new = (y - (torch.einsum("ngb,gbc->ngc", y, gram) - cross) / L).clamp_min(0)
        nt = (1 + math.sqrt(1 + 4 * t * t)) / 2
        y = new + (t - 1) / nt * (new - z)
        z = new
        t = nt
        if (it + 1) % 50 == 0:
            grad = torch.einsum("ngb,gbc->ngc", z, gram) - cross
            err = (z - (z - grad).clamp_min(0)).abs().amax(-1) / cross.abs().amax(
                -1
            ).clamp_min(1e-12)
            if float(err.max()) <= tolerance * 0.5:
                break
    B64 = B.double()
    G = B64 @ B64.transpose(1, 2)
    C = (x.double() @ B64.reshape(-1, d).T).reshape_as(cross)
    z = z.double()

    def error():
        grad = torch.einsum("ngb,gbc->ngc", z, G) - C
        return (z - (z - grad).clamp_min(0)).abs().amax(-1) / C.abs().amax(
            -1
        ).clamp_min(1e-12)

    bad = torch.nonzero(error() > tolerance)
    if len(bad):
        from scipy.optimize import nnls

        bb = B64.cpu().numpy()
        xx = x.double().cpu().numpy()
        for i, j in bad.tolist():
            a, _ = nnls(bb[j].T, xx[i], maxiter=20000)
            z[i, j] = torch.as_tensor(a, device=z.device)
    err = error()
    if (
        not torch.isfinite(z).all()
        or bool((z < 0).any())
        or float(err.max()) > tolerance
    ):
        raise RuntimeError("SLiCS KKT verification failed")
    return z, dict(
        cpu_fallback_rows=len(bad), max_relative_kkt=float(err.max()), iterations=it + 1
    )


class BSFUnits:
    interface = "gated BSF coefficient norm and decoded contribution"

    def __init__(self, model, groups, device):
        self.model = model
        self.groups = list(groups)
        self.device = device
        self.D = model.atoms().detach()[self.groups]

    @torch.inference_mode()
    def maps(self, x):
        V, P, d = x.shape
        z = self.model.encode(
            torch.as_tensor(x.copy(), device=self.device).reshape(-1, d)
        )[:, self.groups]
        score = z.norm(dim=-1).reshape(V, P, -1)
        part = torch.einsum("pgr,grd->pgd", z, self.D).reshape(
            V, P, len(self.groups), d
        )
        return score.cpu().numpy(), part.cpu().numpy()


class SLiCSUnits:
    interface = (
        "independent native per-concept NNLS projection; not a joint additive component"
    )

    def __init__(self, D, slots, device):
        self.device = device
        self.D = D.detach().reshape(75, 12, -1)[list(slots)].to(device)
        self.last_solver = {}

    @torch.inference_mode()
    def maps(self, x):
        V, P, d = x.shape
        z, self.last_solver = partial_nnls(
            torch.as_tensor(x.copy(), device=self.device).reshape(-1, d), self.D
        )
        score = z.norm(dim=-1).reshape(V, P, -1)
        part = torch.einsum("pgr,grd->pgd", z, self.D.double()).reshape(
            V, P, len(self.D), d
        )
        return score.cpu().numpy(), part.cpu().numpy()


class ConceptUnits:
    interface = (
        "native mask-times-presence response and category-token additive contribution"
    )

    def __init__(self, model, slots, device):
        self.model = model
        self.slots = list(slots)
        self.device = device

    @torch.inference_mode()
    def maps(self, x):
        scores = []
        parts = []
        for lo in range(0, len(x), 8):
            xx = torch.as_tensor(x[lo : lo + 8].copy(), device=self.device)
            mask, presence = self.model.concept_model.tokenizer(xx)
            gated = mask * presence
            # Keep the actual (possibly biased/nonlinear) released projectors.
            cp = torch.stack(
                [
                    self.model.concept_model.output_projectors[j](gated[:, j])
                    for j in self.slots
                ],
                1,
            )
            A = self.model.concept_model.token_aggregator.weight[:, self.slots]
            parts.append(torch.einsum("pg,vgd->vpgd", A, cp).cpu().numpy())
            scores.append(gated[:, self.slots].transpose(1, 2).cpu().numpy())
        return np.concatenate(scores), np.concatenate(parts)


def summarize(x, fraction, score, part):
    """Common foreground pooling; keep native response scores unmodified."""
    if (
        x.ndim != 3
        or part.shape[:2] != x.shape[:2]
        or part.shape[-1] != x.shape[-1]
        or score.shape != part.shape[:3]
    ):
        raise ValueError("Native observation shape")
    f = np.asarray(fraction, float)
    fg = f / f.sum(1, keepdims=True)
    bg = (f <= 0.1).astype(float)
    bg /= bg.sum(1, keepdims=True)
    return dict(
        scores=np.einsum("vp,vpg->vg", fg, score),
        vectors=np.einsum("vp,vpgd->vgd", fg, part),
        contrasts=np.einsum("vp,vpgd->vgd", fg - bg, part),
        raw_contrast=np.einsum("vp,vpd->vd", fg - bg, x),
    )
