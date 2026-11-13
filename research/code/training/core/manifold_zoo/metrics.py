import itertools
import numpy as np
from .data import factor


def correlation_match(pack, z, m):
    masks = np.zeros((len(z), m))
    masks[np.arange(len(z))[:, None], pack["ids"]] = 1
    on = (np.linalg.norm(z, axis=2) > 1e-6).astype(float)
    a = masks - masks.mean(0)
    b = on - on.mean(0)
    corr = (
        a.T
        @ b
        / np.maximum(
            np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :],
            1e-12,
        )
    )
    return corr


def r2(target, pred):
    return float(
        1
        - np.sum((target - pred) ** 2)
        / max(np.sum((target - target.mean(0)) ** 2), 1e-20)
    )


def evaluate(cal, test, truth, zc, zt, D, train_mean, support_limit=8, heartbeat=None):
    """Raw factor contributions; no fitted readout or prediction centering.

    Both paper-style test-mask matching and calibration-locked matching are
    reported. Splitting uses an explicitly bounded calibration shortlist;
    failure within that shortlist does NOT prove global unrecoverability.
    """
    m = len(truth["kinds"])
    cc = correlation_match(cal, zc, m)
    ct = correlation_match(test, zt, m)
    rows = []
    g = D.shape[0]
    for f, kind in enumerate(truth["kinds"]):
        if heartbeat and f % 16 == 0:
            heartbeat(f, len(truth["kinds"]))
        ci, targetc = factor(cal, truth, f)
        ti, targett = factor(test, truth, f)
        if len(ci) < 2 or len(ti) < 2:
            rows.append(dict(factor=f, kind=kind, evaluable=False))
            continue
        cg = int(cc[f].argmax())
        tg = int(ct[f].argmax())
        cp = zc[ci, cg] @ D[cg]
        tp = zt[ti, cg] @ D[cg]
        native = zt[ti, tg] @ D[tg]
        shortlist = np.argsort(-cc[f])[: min(support_limit, g)]
        chosen = []
        pred = np.zeros_like(targetc)
        curve = []
        testcurve = []
        for size in range(len(shortlist)):
            options = [h for h in shortlist if h not in chosen]
            h = max(options, key=lambda h: r2(targetc, pred + zc[ci, h] @ D[h]))
            chosen.append(int(h))
            pred += zc[ci, h] @ D[h]
            curve.append(r2(targetc, pred))
            testcurve.append(r2(targett, sum(zt[ti, h] @ D[h] for h in chosen)))
        qualifying = next(
            (
                i + 1
                for i, (a, b) in enumerate(zip(curve, testcurve))
                if a > 0.8 and b > 0.8
            ),
            None,
        )
        off = np.setdiff1d(np.arange(len(zt)), ti)
        rows.append(
            dict(
                factor=f,
                kind=kind,
                evaluable=True,
                calibration_group=cg,
                paper_test_group=tg,
                paper_test_matched_r2=r2(targett, native),
                heldout_matched_r2=r2(targett, tp),
                best_single_in_shortlist_r2=max(
                    r2(targett, zt[ti, h] @ D[h]) for h in shortlist
                ),
                blocks_needed_shortlist=qualifying,
                calibration_support_order=chosen,
                test_support_curve=testcurve,
                restricted_capture_at_true_ambient_rank=testcurve[
                    min(len(testcurve), int(np.linalg.matrix_rank(truth["frames"][f])))
                    - 1
                ],
                off_factor_energy=float(np.mean((zt[off, cg] @ D[cg]) ** 2))
                if len(off)
                else 0.0,
            )
        )
    ok = [r for r in rows if r.get("evaluable")]
    reconstruction = np.einsum("ngr,grd->nd", zt, D)
    ve = float(
        1
        - np.sum((test["x"] - reconstruction) ** 2)
        / max(np.sum((test["x"] - train_mean) ** 2), 1e-20)
    )
    vals = np.array([r["heldout_matched_r2"] for r in ok])
    needed = [r["blocks_needed_shortlist"] for r in ok]
    return dict(
        factors=rows,
        evaluable_factors=len(ok),
        ve=ve,
        paper_mean_r2=float(np.mean([r["paper_test_matched_r2"] for r in ok])),
        heldout_mean_r2=float(vals.mean()),
        fraction_r2_gt08=float(np.mean(vals > 0.8)),
        fraction_r2_gt09=float(np.mean(vals > 0.9)),
        one_block_fraction=float(np.mean([n == 1 for n in needed])),
        multiple_block_fraction=float(
            np.mean([n is not None and n > 1 for n in needed])
        ),
        not_recovered_in_shortlist_fraction=float(np.mean([n is None for n in needed])),
        mean_active_groups=float(np.mean((np.linalg.norm(zt, axis=2) > 1e-6).sum(1))),
        split_measure="Greedy calibration-selected supports among 8 firing-correlated blocks; bounded diagnostic, not an exact minimum over all blocks.",
        paper90_comparison="Thresholds .8 and .9 both reported; the analysis paper does not unambiguously define the cutoff behind its 90% statement.",
    )
