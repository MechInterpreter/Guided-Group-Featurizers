"""Combine the complete fingerprint and controlled-appearance results."""

from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence/fingerprints"
I = ROOT / "figure_sources/inputs/controlled_robustness"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
for name, digest in json.loads((I / "INPUT_SHA256.json").read_text()).items():
    assert hashlib.sha256((I / name).read_bytes()).hexdigest() == digest, name
methods = ["agf_g", "supervised_vanilla", "released_vanilla", "slics", "concept_sae"]
names = ["GGF", "Supervised Vanilla", "Vanilla BSF", "SLiCS", "Concept-SAE"]
colors = dict(
    agf_g="#008c82",
    supervised_vanilla="#c28d29",
    released_vanilla="#78878d",
    slics="#4778b4",
    concept_sae="#8a58ac",
)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
fig = plt.figure(figsize=(6.875, 2.75))
gs = fig.add_gridspec(1, 5, left=0.073, right=0.924, bottom=0.55, top=0.92, wspace=0.35)
cats = json.loads((E / "protocol.json").read_text())["category_names"]
ticks = [0, 24, 49, 74]
for ax, method, name in zip(
    [fig.add_subplot(gs[0, j]) for j in range(5)], methods, names
):
    matrices = []
    for seed in [21, 22, 23]:
        a = np.load(
            ROOT
            / "figure_sources/inputs/fingerprint_plot_arrays"
            / f"{method}_s{seed}.npz"
        )
        uplift = np.maximum(
            0,
            (a["ap"] - a["prevalence"][:, :, None]) / (1 - a["prevalence"][:, :, None]),
        )
        matrices.append(uplift.mean(0)[:, a["primary_units"]].T)
    im = ax.imshow(
        np.mean(matrices, axis=0),
        vmin=0,
        vmax=1,
        cmap="viridis",
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(name, fontsize=7.1, pad=3, fontweight="bold")
    ax.set_xticks(ticks, [t + 1 for t in ticks], fontsize=6)
    ax.set_yticks(
        ticks, [t + 1 for t in ticks] if method == "agf_g" else [], fontsize=6
    )
    ax.tick_params(length=2, pad=1)
cax = fig.add_axes([0.943, 0.58, 0.012, 0.31])
cb = fig.colorbar(im, cax=cax)
cb.ax.tick_params(labelsize=6, length=2, pad=1)
cb.set_label("AP uplift", fontsize=6, labelpad=2)
fig.text(
    0.073, 0.983, "(a) Whole-group category fingerprints", fontsize=8, fontweight="bold"
)
fig.text(0.5, 0.503, "Category index (shared order)", ha="center", fontsize=6.4)

c = pd.read_csv(I / "condition_metrics.csv").dropna(subset=["category_AP"])
per = (
    c.groupby(["model", "seed", "attribute", "level"]).category_AP.mean().reset_index()
)
cap = pd.read_csv(I / "capture_by_category.csv")
cap = cap[
    (cap.group_budget == 1) & cap.attribute.isin(["rotation", "scale", "brightness"])
]
cp = cap.groupby(["model", "seed", "attribute"]).capture.mean().reset_index()
gs = fig.add_gridspec(
    1,
    4,
    left=0.073,
    right=0.988,
    bottom=0.163,
    top=0.395,
    wspace=0.31,
    width_ratios=[1, 1, 1, 1.12],
)
axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
order = ["released_vanilla", "slics", "concept_sae", "supervised_vanilla", "agf_g"]
for ax, attr, label in zip(
    axes[:3],
    ["rotation", "scale", "brightness"],
    ["Rotation (degrees)", "Object scale", "Object brightness"],
):
    for method in order:
        a = (
            per[(per.model == method) & (per.attribute == attr)]
            .groupby("level")
            .category_AP.agg(["mean", "std"])
        )
        x = a.index.to_numpy()
        y = 100 * a["mean"].to_numpy()
        sd = 100 * a["std"].to_numpy()
        ax.plot(
            x,
            y,
            color=colors[method],
            marker="o" if method == "agf_g" else None,
            ms=2,
            lw=1.2 if method == "agf_g" else 0.9,
            ls="--" if method == "supervised_vanilla" else "-",
            label=names[methods.index(method)],
        )
        ax.fill_between(x, y - sd, y + sd, color=colors[method], alpha=0.12, lw=0)
    ax.set_xlabel(label, labelpad=2, fontsize=6.5)
    ax.set_ylim(40, 101)
    ax.set_yticks([40, 60, 80, 100])
    ax.tick_params(labelsize=6.3, length=2, pad=1)
    ax.grid(axis="y", alpha=0.13, lw=0.5)
    if attr == "rotation":
        ax.set_ylabel("Category AP (%)", fontsize=6.7, labelpad=2)
    else:
        ax.set_yticklabels([])
ax = axes[3]
for j, method in enumerate(["released_vanilla", "supervised_vanilla", "agf_g"]):
    a = (
        cp[cp.model == method]
        .groupby("attribute")
        .capture.agg(["mean", "std"])
        .reindex(["rotation", "scale", "brightness"])
    )
    ax.bar(
        np.arange(3) + (j - 1) * 0.23,
        100 * a["mean"],
        0.21,
        color=colors[method],
        yerr=100 * a["std"],
        capsize=1.5,
        error_kw={"elinewidth": 0.55, "capthick": 0.55},
    )
ax.set_xticks(range(3), ["Rot.", "Scale", "Bright."])
ax.set_ylim(0, 21)
ax.set_yticks([0, 10, 20])
ax.tick_params(labelsize=6.3, length=2, pad=1)
ax.set_ylabel("Capture (%)", fontsize=6.7, labelpad=2)
ax.grid(axis="y", alpha=0.13, lw=0.5)
ax.set_axisbelow(True)
fig.text(
    0.073,
    0.438,
    "(b) Recognition across transformations",
    fontsize=8,
    fontweight="bold",
)
fig.text(0.79, 0.438, "(c) Regional Capture", fontsize=8, fontweight="bold")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=5,
    frameon=False,
    fontsize=6.5,
    loc="lower center",
    bbox_to_anchor=(0.53, 0.001),
    handlelength=1.5,
    columnspacing=1.0,
)
for ext in ["pdf", "png"]:
    fig.savefig(
        F / f"ggf_fingerprints_robustness.{ext}",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
plt.close(fig)
print(
    "Combined all five methods, complete transformation curves, Capture and sample-SD displays."
)
