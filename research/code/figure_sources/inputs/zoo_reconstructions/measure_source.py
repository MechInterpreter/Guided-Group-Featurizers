from pathlib import Path
import sys, json, hashlib, time, argparse

H = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(
    description="Read-only seed-0 Zoo reconstruction figure inference; never trains or writes source checkpoints."
)
ap.add_argument("--checkpoint-dir", type=Path, default=H / "downloads")
ap.add_argument(
    "--core-root", type=Path, default=H.parent / "ggf_focused_confirmation/core"
)
ap.add_argument("--wrapper-root", type=Path, required=True)
ap.add_argument("--output", type=Path, default=H / "measured")
args = ap.parse_args()
sys.path.insert(0, str(H.parent / "ggf_focused_confirmation/numpy213"))
sys.path.insert(0, str(args.core_root))
sys.path.insert(0, str(args.wrapper_root))
import numpy as np, torch
from threadpoolctl import threadpool_limits
from zoo_benchmark import data, metrics
from bsf import VanillaBSF, GrassmannianBSF, GroupLassoBSF
from coco_baselines.training import weight_sha

torch.set_num_threads(2)
threadpool_limits(2)
ref = json.loads((args.wrapper_root / "zoo_reference.json").read_text())
cfg = ref["config"]
out = args.output
out.mkdir(exist_ok=True, parents=True)
for r in json.loads((args.checkpoint_dir / "provenance.json").read_text()):
    p = args.checkpoint_dir / f"{r['name']}.pt"
    if p.exists():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == r["metadata"]["sha256"]
print("Generating fixed benchmark locally; NumPy", np.__version__, flush=True)
ds = data.generate(cfg, "goodfire", "independent", 701)
print("Dataset generated", flush=True)
audit = {
    "numpy": np.__version__,
    "torch": torch.__version__,
    "data": data.audit(ds),
    "original_fingerprints": ref["original_data_fingerprints"],
    "regeneration_note": "Same generator, seeds, geometry and split sizes. Floating-point platform differences prevent byte-identical regeneration; comparison below checks original seed-0 scores. Used for visualization only; published metric tables unchanged.",
    "model_seed": 0,
    "factor_selection": "First indexed factor of each non-segment family, independent of all model scores.",
    "point_selection": "Uniform fixed sample, seed 48714, at most 700 present-factor observations. Gate-zero points retained.",
}
points = {}
summaries = {}
for name, cls in [
    ("vanilla", VanillaBSF),
    ("grass", GrassmannianBSF),
    ("lasso", GroupLassoBSF),
    ("ggf", VanillaBSF),
]:
    state = torch.load(
        args.checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=False
    )
    weights = state["best"] if name == "ggf" else state["weights"]
    if name == "ggf":
        assert (
            weight_sha(weights)
            == "e84e92c3930957586a0617c65418360c9501f8fb76ac9abb3d01882768743c1d"
        )
    else:
        assert (
            state["seed"] == 0
        )  # Payload SHA was verified above; historical tensor-hash convention differs.
    model = (
        cls(128, 256, 4, target_l0=4) if name == "lasso" else cls(128, 256, 4, 4)
    ).eval()
    model.load_state_dict(weights)

    def encode(x):
        with torch.no_grad():
            return np.concatenate(
                [
                    model.encode(torch.as_tensor(x[i : i + 2048])).numpy()
                    for i in range(0, len(x), 2048)
                ]
            )

    print("Encoding", name, flush=True)
    zc = encode(ds["packs"]["calibration"]["x"])
    zt = encode(ds["packs"]["test"]["x"])
    D = model.atoms().detach().numpy()
    mapping = metrics.correlation_match(ds["packs"]["calibration"], zc, 128).argmax(1)
    rows = []
    for f, kind in enumerate(ds["truth"]["kinds"]):
        ix, y = data.factor(ds["packs"]["test"], ds["truth"], f)
        g = int(mapping[f])
        pred = zt[ix, g] @ D[g]
        r2 = metrics.r2(y, pred)
        rows.append(dict(factor=f, kind=kind, group=g, r2=r2, n=len(ix)))
        if kind != "segment" and f == ds["truth"]["kinds"].index(kind):
            rng = np.random.default_rng(48714 + f)
            take = np.sort(rng.choice(len(ix), min(700, len(ix)), replace=False))
            frame = ds["truth"]["frames"][f]
            true3 = y @ frame.T
            pred3 = pred @ frame.T
            points[f"{kind}_truth"] = true3[take]
            points[f"{kind}_{name}"] = pred3[take]
            points[f"{kind}_rows"] = ix[take]
            if name == "ggf":
                assert g == f
    recon = np.einsum("ngr,grd->nd", zt, D, optimize=True)
    ve = 1 - float(
        np.sum((ds["packs"]["test"]["x"] - recon) ** 2)
        / np.sum((ds["packs"]["test"]["x"] - ds["packs"]["train"]["x"].mean(0)) ** 2)
    )
    mean = float(np.mean([r["r2"] for r in rows]))
    summaries[name] = {
        "recovery": mean,
        "ve": ve,
        "weights_sha": weight_sha(weights),
        "rows": rows,
    }
    method = {"grass": "grassmannian", "lasso": "group_lasso"}.get(name, name)
    if name == "ggf":
        expected = (0.982305425761166, 0.9795879349112511)
    else:
        r = next(
            r for r in ref["baseline_rows"] if r["seed"] == 0 and r["method"] == method
        )
        expected = (r["heldout_mean_r2"], r["ve"])
    error = max(abs(mean - expected[0]), abs(ve - expected[1]))
    summaries[name]["archived_score_max_abs_difference"] = error
    print(name, mean, ve, "archived difference", error, flush=True)
    assert error < 2e-6, (name, error, mean, ve, expected)
    del zc, zt, model, recon
audit["models"] = summaries
np.savez_compressed(out / "projected_points.npz", **points)
(out / "audit.json").write_text(json.dumps(audit, indent=2))
print("All saved-score checks passed. Measured points saved.", flush=True)
