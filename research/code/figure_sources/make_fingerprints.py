"""Generate vector manuscript figures from the archived frozen fingerprints."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "evidence/fingerprints"
F = ROOT / "figures"
F.mkdir(exist_ok=True)
METHODS = ["agf_g", "supervised_vanilla", "released_vanilla", "slics", "concept_sae"]
NAMES = ["GGF", "Supervised Vanilla", "Vanilla BSF", "SLiCS", "Concept-SAE"]
COLORS = ["#008f82", "#73b7ad", "#586e85", "#c4933d", "#9770ae"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
p = json.loads((E / "protocol.json").read_text())
names = p["category_names"]
ticks = [0, 24, 49, 74]
matrices = []
for method in METHODS:
    mm = []
    for seed in [21, 22, 23]:
        a = np.load(
            ROOT
            / "figure_sources/inputs/fingerprint_plot_arrays"
            / f"{method}_s{seed}.npz"
        )
        ap = a["ap"]
        prev = a["prevalence"]
        uplift = np.maximum(0, (ap - prev[:, :, None]) / (1 - prev[:, :, None]))
        mm.append(uplift.mean(0)[:, a["primary_units"]].T)
    matrices.append(np.array(mm))
for kind in ["mean", "SD"]:
    fig, axes = plt.subplots(1, 5, figsize=(6.875, 1.78))
    fig.subplots_adjust(left=0.085, right=0.92, bottom=0.28, top=0.86, wspace=0.35)
    vmax = (
        1
        if kind == "mean"
        else np.ceil(max(a.std(0, ddof=1).max() for a in matrices) / 0.05) * 0.05
    )
    for ax, method, name, mm in zip(axes, METHODS, NAMES, matrices):
        v = mm.mean(0) if kind == "mean" else mm.std(0, ddof=1)
        im = ax.imshow(
            v,
            vmin=0,
            vmax=vmax,
            cmap="viridis",
            aspect="equal",
            interpolation="nearest",
            rasterized=True,
        )
        ax.set_title(name, fontsize=7.2, pad=4, fontweight="bold")
        ax.set_xticks(
            ticks, [names[t] for t in ticks], fontsize=5.7, rotation=55, ha="right"
        )
        ax.set_yticks(
            ticks, [names[t] for t in ticks] if method == METHODS[0] else [], fontsize=6
        )
        ax.tick_params(length=2, pad=1)
    cax = fig.add_axes([0.94, 0.31, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=6, length=2, pad=1)
    cb.set_label("AP uplift" if kind == "mean" else "Sample SD", fontsize=6, labelpad=2)
    fig.text(0.5, 0.015, "Category tested", ha="center", fontsize=6.5)
    # The caption explains the row identity, allowing the matrix to stay compact.
    filename = (
        "ggf_category_fingerprints"
        if kind == "mean"
        else "ggf_category_fingerprints_SD"
    )
    fig.savefig(F / f"{filename}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(F / f"{filename}.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

summary = {
    q["method"]: q for q in json.loads((E / "summary.json").read_text())["aggregate"]
}
fig, axes = plt.subplots(1, 2, figsize=(6.875, 2.12))
fig.subplots_adjust(left=0.08, right=0.99, bottom=0.27, top=0.86, wspace=0.25)
for ax, key, title in zip(
    axes,
    ["utility_direction_grouping_lift", "response_direction_grouping_lift"],
    ["Contribution fingerprints", "Response fingerprints"],
):
    for i in range(4):
        a = summary[METHODS[i]][key]
        ax.bar(i, a["mean"], color=COLORS[i], width=0.65)
        ax.errorbar(
            i,
            a["mean"],
            yerr=a["sample_SD"],
            fmt="none",
            ecolor="#183b3a",
            capsize=3,
            lw=0.8,
        )
        ax.scatter(
            i + np.array([-0.10, 0, 0.10]),
            a["seed_values"],
            s=10,
            c="#183b3a",
            zorder=4,
        )
    ax.set_title(title, fontweight="bold", fontsize=8)
    ax.set_xticks(
        range(4), ["GGF", "Supervised\nVanilla", "Vanilla BSF", "SLiCS"], fontsize=7
    )
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Agreement above regrouping control", fontsize=7)
fig.savefig(
    F / "ggf_direction_fingerprint_agreement.pdf", bbox_inches="tight", pad_inches=0.03
)
fig.savefig(
    F / "ggf_direction_fingerprint_agreement.png",
    dpi=500,
    bbox_inches="tight",
    pad_inches=0.03,
)
plt.close(fig)
print("Rendered mean, sample-SD and direction-agreement vector figures.")
