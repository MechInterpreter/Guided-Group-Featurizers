import hashlib
import numpy as np

RANKS = dict(
    segment=1, circle=2, disk=2, sphere=3, torus=3, mobius=3, swiss_roll=3, helix=3
)


def raw(kind, n, rng):
    t = rng.uniform(0, 2 * np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    if kind == "segment":
        return rng.uniform(-1, 1, (n, 1))
    if kind == "circle":
        return np.c_[np.cos(t), np.sin(t)]
    if kind == "disk":
        return np.sqrt(rng.random(n))[:, None] * np.c_[np.cos(t), np.sin(t)]
    if kind == "sphere":
        z = rng.uniform(-1, 1, n)
        r = np.sqrt(1 - z * z)
        return np.c_[r * np.cos(t), r * np.sin(t), z]
    if kind == "torus":
        return np.c_[
            (2 + 0.5 * np.cos(v)) * np.cos(t),
            (2 + 0.5 * np.cos(v)) * np.sin(t),
            0.5 * np.sin(v),
        ]
    if kind == "mobius":
        s = rng.uniform(-0.5, 0.5, n)
        return np.c_[
            (1 + s * np.cos(t / 2)) * np.cos(t),
            (1 + s * np.cos(t / 2)) * np.sin(t),
            s * np.sin(t / 2),
        ]
    if kind == "swiss_roll":
        t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
        return np.c_[t * np.cos(t), rng.uniform(-1, 1, n), t * np.sin(t)]
    if kind == "helix":
        t = rng.uniform(-2 * np.pi, 2 * np.pi, n)
        return np.c_[np.cos(t), np.sin(t), 0.2 * t]
    raise ValueError(kind)


def generate(cfg, suite, condition, seed=701):
    """Compact factor storage: N*4*3 coordinates, never N*128*128 targets."""
    rng = np.random.default_rng(seed)
    curved = (
        ["circle", "helix", "torus", "disk", "swiss_roll"]
        if suite == "analysis"
        else list(RANKS)[1:]
    )
    m = cfg["factors"]
    d = cfg["dimension"]
    k = cfg["active_factors"]
    kinds = ["segment"] * (m // 2) + [
        curved[i % len(curved)] for i in range(m - m // 2)
    ]
    frames = np.zeros((m, 3, d))
    means = []
    scales = []
    for f, kind in enumerate(kinds):
        r = RANKS[kind]
        frames[f, :r] = np.linalg.qr(rng.normal(size=(d, r)))[0].T
        a = raw(kind, cfg["normalization_rows"], rng)
        mu = a.mean(0)
        means.append(mu)
        scales.append(float(np.sqrt(np.mean(np.sum((a - mu) ** 2, 1)))))
    packs = {}
    for si, split in enumerate(["train", "select", "calibration", "test"]):
        r = np.random.default_rng(
            seed
            + 100003 * (si + 1)
            + (cfg.get("test_seed_offset", 0) if split == "test" else 0)
        )
        n = cfg[split + "_rows"]
        ids = np.empty((n, k), np.int16)
        for lo in range(0, n, 4096):
            hi = min(lo + 4096, n)
            ids[lo:hi] = np.argpartition(r.random((hi - lo, m)), k - 1, axis=1)[:, :k]
        if condition == "correlated":
            # Fixed-size paired co-occurrence sampling; unspecified in the reference paper.
            rows = np.flatnonzero(r.random(n) < 0.5)
            pair = r.integers(m // 2, size=len(rows)) * 2
            for j, row in enumerate(rows):
                p = [int(pair[j]), int(pair[j] + 1)]
                rest = r.choice(np.setdiff1d(np.arange(m), p), k - 2, replace=False)
                ids[row] = np.r_[p, rest]
        coords = np.zeros((n, k, 3), np.float64)
        x = np.zeros((n, d), np.float64)
        for f, kind in enumerate(kinds):
            rows, slots = np.where(ids == f)
            b = (raw(kind, len(rows), r) - means[f]) / scales[f]
            coords[rows, slots, : b.shape[1]] = b
            x[rows] += b @ frames[f, : b.shape[1]]
        packs[split] = dict(x=x, ids=ids, coords=coords)
    # Appendix H: unit mean squared activation norm. No extra global translation.
    scale = float(np.sqrt(np.mean(np.sum(packs["train"]["x"] ** 2, 1))))
    for p in packs.values():
        p["x"] = (p["x"] / scale).astype("float32")
        p["coords"] = (p["coords"] / scale).astype("float32")
    truth = dict(
        frames=frames, kinds=kinds, mean=means, scales=scales, global_scale=scale
    )
    fingerprints = {
        s: hashlib.sha256(
            p["x"].tobytes() + p["ids"].tobytes() + p["coords"].tobytes()
        ).hexdigest()
        for s, p in packs.items()
    }
    return dict(
        packs=packs,
        truth=truth,
        fingerprints=fingerprints,
        suite=suite,
        condition=condition,
        data_seed=seed,
    )


def factor(pack, truth, f):
    rows, slots = np.where(pack["ids"] == f)
    return rows, pack["coords"][rows, slots].astype(float) @ truth["frames"][f]


def subset(pack, n):
    return {k: v[:n] for k, v in pack.items()}


def audit(dataset):
    p = dataset["packs"]
    t = dataset["truth"]
    errors = []
    for split, pack in p.items():
        approx = np.einsum(
            "nkr,nkrd->nd", pack["coords"][:256], t["frames"][pack["ids"][:256]]
        )
        errors.append(float(abs(approx - pack["x"][:256]).max()))
        assert np.isfinite(pack["x"]).all() and np.isfinite(pack["coords"]).all()
        assert all(len(set(row)) == len(row) for row in pack["ids"][:2000])
    energy = float(np.mean(np.sum(p["train"]["x"].astype(float) ** 2, 1)))
    assert max(errors) < 1e-6 and abs(energy - 1) < 1e-6
    return dict(
        factor_sum_max_error=max(errors),
        train_squared_norm=energy,
        fingerprints=dataset["fingerprints"],
        global_centering=False,
        one_training_scale_for_all_splits=True,
        correlation_recipe_exactly_published=dataset["condition"] != "correlated",
    )
