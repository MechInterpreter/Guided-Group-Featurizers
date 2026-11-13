"""Evaluation-only NNLS acceleration; every row requires the original certificate.

GPU FISTA is only a warm start. Cached float64 CPU active-set refinement can
add ANY atom, and full scipy.nnls remains the terminal fallback. No ridge,
Top-k, labels, support restriction, new readout, or model update is introduced.
"""

import math, time, warnings
import numpy as np
import torch
from scipy.linalg import cho_factor, cho_solve, lstsq, LinAlgError, LinAlgWarning
from scipy.optimize import nnls


def certificate(z, x, D):
    z = z.double()
    x = x.double()
    D = D.double()
    cross = x @ D.T
    grad = (z @ D - x) @ D.T
    return (z - (z - grad).clamp_min(0)).abs().amax(1) / cross.abs().amax(1).clamp_min(
        1e-12
    )


class SharedActiveSet:
    """Lawson-Hanson feasibility corrections with a reusable Gram matrix.

    Supports are working sets, never final constraints. Original-space KKT
    verification plus full-dictionary fallback guards normal-equation error.
    """

    def __init__(self, D, tolerance=1e-5, maxiter=20000):
        self.D = np.array(D, dtype=np.float64, order="C", copy=True)
        self.A = np.asfortranarray(self.D.T)
        self.G = self.D @ self.D.T
        self.tolerance = tolerance
        self.maxiter = maxiter
        self.fallbacks = 0

    def error(self, z, b):
        c = self.D @ b
        g = self.D @ (z @ self.D - b)
        return np.max(np.abs(z - np.maximum(z - g, 0))) / max(np.max(np.abs(c)), 1e-12)

    def _least_squares(self, ids, c, b):
        G = self.G[np.ix_(ids, ids)]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                result = cho_solve(
                    cho_factor(G, check_finite=False), c[ids], check_finite=False
                )
            # A residual test alone is insufficient for ill-conditioned Gram
            # systems. The original-space certificate is checked after every
            # accepted iterate; singular systems use original-space QR here.
            if not np.isfinite(result).all():
                raise LinAlgError("nonfinite solution")
            if np.max(np.abs(G @ result - c[ids])) > 1e-9 * max(
                np.max(np.abs(c[ids])), 1.0
            ):
                raise LinAlgError("normal-equation residual")
            return result
        except (LinAlgError, LinAlgWarning):
            return lstsq(
                self.A[:, ids], b, cond=None, check_finite=False, lapack_driver="gelsy"
            )[0]

    def solve(self, b, start):
        b = np.asarray(b, dtype=np.float64)
        z = np.maximum(np.asarray(start, dtype=np.float64), 0).copy()
        if z.shape != (len(self.D),) or not np.isfinite(z).all():
            raise ValueError("Invalid NNLS warm start")
        if self.error(z, b) <= self.tolerance:
            return z, False
        # Discarding tiny warm-start entries is only initialization. Every
        # excluded coordinate is eligible for re-entry via the full gradient.
        floor = 1e-8 * max(float(z.max()), 1e-12)
        z[z <= floor] = 0
        passive = z > 0
        c = self.D @ b
        for _ in range(self.maxiter):
            ids = np.flatnonzero(passive)
            if len(ids):
                u = np.zeros_like(z)
                u[ids] = self._least_squares(ids, c, b)
                negative = passive & (u <= 0)
                if negative.any():
                    denom = z[negative] - u[negative]
                    valid = denom > 0
                    alpha = (
                        float(np.min(z[negative][valid] / denom[valid]))
                        if valid.any()
                        else 0.0
                    )
                    z = np.maximum(z + alpha * (u - z), 0)
                    boundary = passive & (z <= 1e-13 * max(float(z.max()), 1.0))
                    if not boundary.any():
                        break
                    passive[boundary] = False
                    z[boundary] = 0
                    continue
                z = u
            if self.error(z, b) <= self.tolerance:
                return z, False
            # Never restrict inference to the current passive set.
            w = self.D @ (b - z @ self.D)
            w[passive] = -np.inf
            candidate = int(np.argmax(w))
            if not np.isfinite(w[candidate]) or w[candidate] <= 0:
                break
            passive[candidate] = True
        self.fallbacks += 1
        answer, _ = nnls(self.A, b, maxiter=self.maxiter)
        if self.error(answer, b) > self.tolerance:
            raise RuntimeError("Full NNLS fallback failed unchanged KKT certificate")
        return answer, True


class GPUJointModel:
    def __init__(self, D, cfg, device="cuda", iterations=2000, check_every=100):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Select a GPU runtime")
        if D.ndim != 2 or not torch.isfinite(D).all():
            raise ValueError("Finite matrix required")
        if cfg["nnls_tolerance"] != 1e-5 or cfg["nnls_fallback_iterations"] != 20000:
            raise ValueError("Pinned NNLS tolerance/maxiter changed")
        self.D = D.detach().to(device=self.device, dtype=torch.float64).clone()
        self.cfg = cfg
        self.d = D.shape[1]
        self.group_size = 12
        self.n_groups = len(D) // 12
        self.D32 = self.D.float()
        self.G = self.D32 @ self.D32.T
        self.L = float(torch.linalg.eigvalsh(self.G).max().clamp_min(1e-8)) * 1.00001
        self.active = SharedActiveSet(
            self.D.cpu().numpy(), cfg["nnls_tolerance"], cfg["nnls_fallback_iterations"]
        )
        self.iterations = iterations
        self.check_every = check_every
        self.last_stats = {}

    @torch.no_grad()
    def warm_start(self, x):
        xf = x.float()
        cross = xf @ self.D32.T
        pending = torch.arange(len(x), device=self.device)
        z = torch.zeros_like(cross)
        y = z.clone()
        out = torch.zeros(
            (len(x), len(self.D)), device=self.device, dtype=torch.float64
        )
        t = 1.0
        used = 0
        for k in range(self.iterations):
            new = (y - (y @ self.G - cross) / self.L).clamp_min(0)
            nt = (1 + math.sqrt(1 + 4 * t * t)) / 2
            y = new + (t - 1) / nt * (new - z)
            z = new
            t = nt
            used = k + 1
            if used % self.check_every == 0 or used == self.iterations:
                ok = certificate(z, x[pending], self.D) <= self.cfg["nnls_tolerance"]
                out[pending[ok]] = z[ok].double()
                keep = ~ok
                if not bool(keep.any()):
                    return out, dict(gpu_certified=len(x), pending=0, iterations=used)
                pending = pending[keep]
                z = z[keep]
                y = y[keep]
                cross = cross[keep]
            # Fixed numerical restart helps stalled accelerated iterations.
            if used % 500 == 0:
                y = z.clone()
                t = 1.0
        out[pending] = z.double()
        return out, dict(
            gpu_certified=len(x) - len(pending), pending=len(pending), iterations=used
        )

    @torch.no_grad()
    def solve(self, x):
        started = time.monotonic()
        x = x.to(device=self.device)
        if (
            x.ndim != 2
            or x.shape[1] != self.d
            or len(x) == 0
            or not torch.isfinite(x).all()
        ):
            raise ValueError("Finite nonempty matching patches required")
        z, stats = self.warm_start(x)
        bad = torch.nonzero(
            certificate(z, x, self.D) > self.cfg["nnls_tolerance"]
        ).flatten()
        # Transfer only once per batch; matrices are cached once per model.
        if len(bad):
            bx = x[bad].double().cpu().numpy()
            bz = z[bad].cpu().numpy()
            answers = []
            fallbacks = 0
            for row, start in zip(bx, bz):
                answer, fallback = self.active.solve(row, start)
                answers.append(answer)
                fallbacks += int(fallback)
            z[bad] = torch.as_tensor(
                np.asarray(answers), device=self.device, dtype=torch.float64
            )
            stats["full_scipy_fallbacks"] = fallbacks
        error = certificate(z, x, self.D)
        if (
            not torch.isfinite(z).all()
            or bool((z < 0).any())
            or float(error.max()) > self.cfg["nnls_tolerance"]
        ):
            raise RuntimeError(
                "Accelerated NNLS failed original per-row float64 KKT certificate; no image committed"
            )
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.last_stats = dict(
            stats,
            rows=len(x),
            max_relative_kkt=float(error.max()),
            seconds=time.monotonic() - started,
        )
        return z


def benchmark(model, x):
    """Fixed eight patches; numerical equivalence and timing, not model selection."""
    x = x[:8].to(model.device)
    if len(x) != 8:
        raise ValueError("Eight fixed benchmark patches required")
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.monotonic()
    z = model.solve(x)
    if model.device.type == "cuda":
        torch.cuda.synchronize()
    accelerated = time.monotonic() - start
    xx = x.double().cpu().numpy()
    start = time.monotonic()
    baseline = np.stack(
        [
            nnls(model.active.A, b, maxiter=model.cfg["nnls_fallback_iterations"])[0]
            for b in xx
        ]
    )
    baseline_seconds = time.monotonic() - start
    zz = z.cpu().numpy()
    D = model.active.D
    current = np.square(zz @ D - xx).sum(1)
    reference = np.square(baseline @ D - xx).sum(1)
    scaled = np.abs(current - reference) / np.maximum(np.square(xx).sum(1), 1.0)
    if np.max(scaled) > 1e-7:
        raise RuntimeError(
            "Acceleration benchmark disagrees with reference NNLS objective; no evaluation image committed"
        )
    ref_error = np.max([model.active.error(a, b) for a, b in zip(baseline, xx)])
    if ref_error > model.cfg["nnls_tolerance"]:
        raise RuntimeError("Reference NNLS failed original KKT check")
    return dict(
        patches=8,
        accelerated_seconds=accelerated,
        reference_seconds=baseline_seconds,
        observed_probe_speedup=baseline_seconds / max(accelerated, 1e-12),
        max_scaled_SSE_difference=float(np.max(scaled)),
        max_relative_kkt=model.last_stats["max_relative_kkt"],
        reference_kkt=float(ref_error),
        caveat="Small timing probe excludes dictionary initialization and does not predict full-run speed; no model/settings selected.",
        solver_stats=model.last_stats,
    )
