from pathlib import Path
import json, shutil, hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

H = Path(__file__).resolve().parent


def main():
    staged = (H / "measured/projected_points.npz").exists()
    inputs = H / "measured" if staged else H / "inputs/zoo_reconstructions"
    p = inputs / "projected_points.npz"
    a = json.loads(
        (inputs / ("audit.json" if staged else "inference_audit.json")).read_text()
    )
    z = np.load(p)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    kinds = ["torus", "sphere", "disk", "mobius", "swiss_roll", "helix", "circle"]
    titles = [
        "Ground\ntruth",
        "Vanilla\nBSF",
        "Grassmannian\nBSF",
        "Group Lasso\nBSF",
        "GGF",
    ]
    keys = [None, "vanilla", "grass", "lasso", "ggf"]
    palette = ["#6f7f90", "#7e57a3", "#7e57a3", "#7e57a3", "#008b80"]
    labels = {
        "torus": "Torus",
        "sphere": "Sphere",
        "disk": "Disk",
        "mobius": "Möbius",
        "swiss_roll": "Swiss roll",
        "helix": "Helix",
        "circle": "Circle",
    }
    fig = plt.figure(figsize=(6.875, 6.3), facecolor="white")
    left = 0.095
    right = 0.995
    bottom = 0.023
    top = 0.912
    cw = (right - left) / 5
    rh = (top - bottom) / 7
    for j, title in enumerate(titles):
        fig.text(
            left + (j + 0.5) * cw,
            0.965,
            title,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=palette[j],
        )
    chosen = []
    for i, kind in enumerate(kinds):
        truth = z[kind + "_truth"]
        center = truth.mean(0)
        rms = float(np.sqrt(np.mean(np.sum((truth - center) ** 2, axis=1))))
        arrays = [
            (z[kind + "_truth" if k is None else kind + "_" + k] - center) / rms
            for k in keys
        ]
        # One transform and one common display range, including every prediction.
        span = np.max(np.abs(np.concatenate(arrays)), axis=0)
        radius = max(span.max() * 1.06, 1.0)
        # The true azimuth provides point correspondence; predictions never choose colors.
        colors = plt.get_cmap("hsv")(
            (
                np.arctan2(truth[:, 1] - center[1], truth[:, 0] - center[0])
                / (2 * np.pi)
                + 1
            )
            % 1
        )
        y = top - (i + 1) * rh
        fig.text(
            0.011,
            y + rh * 0.53,
            labels[kind],
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#384453",
        )
        f = next(r["factor"] for r in a["models"]["ggf"]["rows"] if r["kind"] == kind)
        row = {
            "kind": kind,
            "factor": f,
            "points": len(truth),
            "common_center": center.tolist(),
            "common_rms_scale": rms,
            "common_radius": float(radius),
            "models": {},
        }
        for j, (key, pts) in enumerate(zip(keys, arrays)):
            ax = fig.add_axes(
                [left + j * cw, y + 0.018, cw, rh - 0.012],
                projection="3d",
                computed_zorder=False,
            )
            ax.set_proj_type("ortho")
            ax.view_init(elev=24, azim=-56)
            ax.set_box_aspect((1, 1, 0.82))
            ax.set_axis_off()
            floor = -radius * 1.05
            v = [
                [-radius, -radius, floor],
                [radius, -radius, floor],
                [radius, radius, floor],
                [-radius, radius, floor],
            ]
            ax.add_collection3d(
                Poly3DCollection([v], facecolors="#eff2f6", edgecolors="none", zorder=0)
            )
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                np.full(len(pts), floor + 0.004),
                c="#5e6773",
                s=0.34,
                alpha=0.11,
                edgecolors="none",
                rasterized=True,
                zorder=1,
            )
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                pts[:, 2],
                c=colors,
                s=1.0,
                alpha=0.9,
                edgecolors="none",
                depthshade=False,
                rasterized=True,
                zorder=2,
            )
            ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(floor, radius))
            if key:
                r = next(r for r in a["models"][key]["rows"] if r["factor"] == f)
                fig.text(
                    left + (j + 0.5) * cw,
                    y + 0.003,
                    f"$R^2$ {100 * r['r2']:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="#008b80" if key == "ggf" else "#46505b",
                )
                row["models"][key] = {"r2_128d": r["r2"], "group": r["group"]}
        chosen.append(row)
    targets = [H / "cvpr2027", H / "preprint"] if staged else [H.parent]
    for target in targets:
        out = target / "figures"
        out.mkdir(exist_ok=True)
        for ext in ["pdf", "png", "svg"]:
            fig.savefig(
                out / f"ggf_zoo_reconstructions.{ext}", dpi=250, facecolor="white"
            )
    plt.close(fig)
    out = H / "cvpr2027/figure_sources/inputs/zoo_reconstructions" if staged else inputs
    out.mkdir(parents=True, exist_ok=True)
    if staged:
        shutil.copy2(p, out / "projected_points.npz")
        shutil.copy2(H / "measured/audit.json", out / "inference_audit.json")
    report = {
        "seed": 0,
        "rows": chosen,
        "columns": titles,
        "coordinate_system": "Ground-truth embedding frame, shared for every method in a row; single ground-truth center and RMS scale. No prediction-specific fit, rotation or rescaling.",
        "colors": "Ground-truth azimuth, identical for paired observations across all columns.",
        "hidden_error": "Projections omit off-factor directions. Full 128D factor R2 is printed; unchanged all-factor three-seed metrics remain in tables.",
        "minimum_label_font_pt": 9,
        "width_inches": 6.875,
        "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
    }
    (out / "plot_audit.json").write_text(json.dumps(report, indent=2))
    if staged:
        shutil.copy2(
            H / "plot.py", H / "cvpr2027/figure_sources/make_zoo_reconstructions.py"
        )
        shutil.copy2(
            H / "measure.py",
            H / "cvpr2027/figure_sources/inputs/zoo_reconstructions/measure_source.py",
        )
        shutil.copy2(
            H / "downloads/provenance.json", out / "checkpoint_provenance.json"
        )
    print(json.dumps(chosen, indent=2))


if __name__ == "__main__":
    main()
