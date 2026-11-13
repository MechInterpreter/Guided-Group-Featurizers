"""Saved three-seed Zoo comparison; illustrative true generator geometries."""

from pathlib import Path
import csv, hashlib, importlib.util, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

R = Path(__file__).resolve().parent.parent
S = R / "evidence/zoo_guided/results"
O = R / "figures"
O.mkdir(exist_ok=True)
source = S / "zoo_complete.json"
d = json.loads(source.read_text())
assert d["complete"]
rows = d["models"]
means = {x["method"]: x for x in d["means"]}
order = ["ggf_selected", "vanilla", "grassmannian", "group_lasso"]
labels = ["GGF", "Vanilla", "Grassmannian", "Group Lasso"]
colors = ["#007F78", "#647D8A", "#A97837", "#80649D"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
    }
)

fig = plt.figure(figsize=(3.25, 1.55), facecolor="white")
axes = [fig.add_axes([0.33, 0.28, 0.25, 0.55]), fig.add_axes([0.69, 0.28, 0.25, 0.55])]
for ax, key, title, xlabel in zip(
    axes,
    ["heldout_mean_r2", "ve"],
    ["Factor recovery", "Reconstruction"],
    ["$R^2$ (%)", "VE (%)"],
):
    for j, (method, color) in enumerate(zip(order, colors)):
        vals = np.array([100 * r[key] for r in rows if r["method"] == method])
        assert len(vals) == 3
        v = vals.mean()
        sd = vals.std(ddof=1)
        ax.barh(j, v, height=0.62, color=color, zorder=2)
        ax.errorbar(
            v,
            j,
            xerr=sd,
            fmt="none",
            ecolor="#203A45",
            elinewidth=0.65,
            capsize=2,
            zorder=3,
        )
        ax.text(
            v - 4,
            j,
            f"{v:.2f}",
            ha="right",
            va="center",
            fontsize=9,
            color="white",
            weight="bold",
            zorder=4,
        )
    ax.set(
        xlim=(0, 103),
        ylim=(3.55, -0.55),
        xticks=[0, 50, 100],
        yticks=range(4),
        yticklabels=labels if ax is axes[0] else [],
        xlabel=xlabel,
    )
    ax.set_title(title, fontsize=9, pad=7, weight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=2, pad=2)
    ax.grid(axis="x", color="#E3E9EB", lw=0.5, zorder=0)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        O / ("ggf_zoo_comparison." + ext), **({"dpi": 240} if ext == "png" else {})
    )
plt.close(fig)

# Draw fresh latent samples from the unchanged generator. These are schematic
# ground-truth families, not learned GGF embeddings or evaluated test points.
p = R / "figure_sources/inputs/zoo_guided/benchmark_data.py"
spec = importlib.util.spec_from_file_location("zoo_benchmark_data", p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
kinds = [
    "segment",
    "circle",
    "disk",
    "sphere",
    "torus",
    "mobius",
    "swiss_roll",
    "helix",
]
display = {
    "segment": "Segment",
    "circle": "Circle",
    "disk": "Disk",
    "sphere": "Sphere",
    "torus": "Torus",
    "mobius": "Möbius strip",
    "swiss_roll": "Swiss roll",
    "helix": "Helix",
}
factors = list(csv.DictReader((S / "zoo_factors.csv").open()))
fig = plt.figure(figsize=(7.0, 3.00), facecolor="white")
rng = np.random.default_rng(48713)
geometry = []
for j, kind in enumerate(kinds):
    ax = fig.add_axes(
        [0.012 + (j % 4) * 0.247, 0.51 - (j // 4) * 0.48, 0.235, 0.42], projection="3d"
    )
    a = module.raw(kind, 700, rng)
    a -= a.mean(0)
    a /= np.sqrt((a * a).sum(1).mean())
    points = np.zeros((len(a), 3))
    points[:, : a.shape[1]] = a
    ax.scatter(*points.T, s=1.3, c="#007F78", alpha=0.58, linewidths=0, depthshade=True)
    ax.set_axis_off()
    ax.view_init(elev=23, azim=-52)
    radius = max(np.abs(points).max(), 0.8) * 1.03
    ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius))
    ax.set_box_aspect([1, 1, 1])
    seed_values = [
        100
        * np.mean(
            [
                float(r["assigned_r2"])
                for r in factors
                if r["method"] == "ggf_selected"
                and r["kind"] == kind
                and int(r["seed"]) == seed
            ]
        )
        for seed in [0, 1, 2]
    ]
    value = float(np.mean(seed_values))
    count = sum(
        r["method"] == "ggf_selected" and r["kind"] == kind and int(r["seed"]) == 0
        for r in factors
    )
    ax.text2D(
        0.5,
        0.97,
        display[kind],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        weight="bold",
    )
    ax.text2D(
        0.5,
        -0.01,
        f"$R^2$ {value:.2f} ± {np.std(seed_values, ddof=1):.3f}%",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="#25434A",
    )
    geometry.append(
        dict(
            kind=kind,
            factors_per_seed=count,
            mean_recovery_pct=value,
            seed_recovery_pct=seed_values,
        )
    )
for ext in ["pdf", "svg", "png"]:
    fig.savefig(
        O / ("ggf_zoo_geometry." + ext), **({"dpi": 240} if ext == "png" else {})
    )
plt.close(fig)

fonts = {}
for name in ["ggf_zoo_comparison", "ggf_zoo_geometry"]:
    with pymupdf.open(O / (name + ".pdf")) as doc:
        sizes = []
        for block in doc[0].get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["text"].strip():
                        assert (doc[0].rect + (-0.1, -0.1, 0.1, 0.1)).contains(
                            pymupdf.Rect(span["bbox"])
                        ), span["text"]
                        if span["text"] not in ["2"]:
                            sizes.append(span["size"])
        fonts[name] = min(sizes)
audit = dict(
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    methods=order,
    seeds=[0, 1, 2],
    error_bars="One sample standard deviation over three model seeds; not confidence intervals.",
    primary_recovery="Calibration-locked one-group R2; fixed assigned group for GGF gives identical values.",
    bar_axes_start_at_zero=True,
    comparison_width_bp=234,
    geometry_width_bp=504,
    geometry_definition="700 fresh points per known factor family from original generator; schematic normalized coordinates, not learned group embeddings or measured test points.",
    geometry_generator_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
    geometry_seed=48713,
    geometry=geometry,
    minimum_main_label_font_pt=fonts,
    full_variance_R2_not_percent_recovered=True,
    selected_GGF_supervision="Known component contributions plus factor presence; stronger than COCO masks.",
)
(R / "figure_sources/inputs/zoo_guided/plot_audit.json").write_text(
    json.dumps(audit, indent=2)
)
print(json.dumps(audit, indent=2))
