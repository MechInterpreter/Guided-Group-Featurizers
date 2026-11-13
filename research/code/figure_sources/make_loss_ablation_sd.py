"""Reproduce the archived loss ablation with saved means and sample SD."""

from pathlib import Path
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent.parent
(R / "figures").mkdir(exist_ok=True)
q = json.loads((R / "figure_sources/inputs/agf_loss_ablation_main.json").read_text())
vals = np.array(q["percentages"])
sd = np.array(q["sample_SD"])
assert vals.shape == sd.shape == (4, 2)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
    }
)
fig, axs = plt.subplots(1, 2, figsize=(2.95, 1.40), layout="constrained")
for j, ax in enumerate(axs):
    for i, color in enumerate(["#527E9D", "#77868C", "#AD761E", "#7951A0"]):
        ax.plot([vals[0, j], vals[i, j]], [i, i], color=color, lw=1.5)
        ax.errorbar(
            vals[i, j],
            i,
            xerr=sd[i, j],
            fmt="o",
            markersize=4,
            color=color,
            ecolor="#26343a",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )
        ax.text(
            vals[i, j] + sd[i, j] + 0.6,
            i,
            f"{vals[i, j]:.2f}",
            va="center",
            fontsize=10,
        )
    ax.set(
        yticks=range(4),
        yticklabels=["Recon.", "+ contrast", "+ detection", "+ both"] if j == 0 else [],
        xlim=(18, 40) if j == 0 else (57, 77),
        xlabel=["Capture (%)", "Patch AP (%)"][j],
    )
    ax.set_ylim(3.55, -0.45)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([20, 30] if j == 0 else [60, 70])
    ax.spines["left"].set_visible(False)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(R / ("figures/agf_loss_lollipop." + ext), dpi=180)
plt.close(fig)
