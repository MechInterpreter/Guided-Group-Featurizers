from pathlib import Path
import json, csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

B = Path(__file__).resolve().parent.parent
OUT = B / "figures"
OUT.mkdir(exist_ok=True)
R = B / "figure_sources/inputs"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "savefig.facecolor": "white",
    }
)
teal = "#007D75"
purple = "#7951A0"
blue = "#527E9D"
amber = "#AD761E"
gray = "#77868C"


def save(fig, name):
    fig.savefig(OUT / (name + ".pdf"))
    fig.savefig(B / (name + ".png"), dpi=180)
    plt.close(fig)


rows = list(csv.DictReader((R / "matched_baselines_means.csv").open()))
new = list(csv.DictReader((B / "evidence/width_balance/three_seed_means.csv").open()))
cap40 = list(csv.DictReader((B / "evidence/cap040_test/three_seed_means.csv").open()))
selected = list(csv.DictReader((B / "evidence/baseline_comparison.csv").open()))
# The family-grouped baseline plot has a separate, evidence-preserving renderer.
import runpy

runpy.run_path(str(B / "figure_sources/make_grouped_baselines.py"), run_name="__main__")
colors = [blue, gray, amber, purple]

fig, ax = plt.subplots(figsize=(3.0, 2.0), layout="constrained")
positions = [(25, 65.25), (28, 67.65), (26.5, 63.5), (34, 66.5), (19.8, 67.5)]
names = ["3D fixed", "4D fixed", "3D adaptive", "4D adaptive", "Vanilla 4D"]
for r, name, color, marker, position in zip(
    new, names, [purple, blue, amber, teal, gray], ["o", "s", "^", "D", "X"], positions
):
    x = float(r["capture_pct"])
    y = float(r["VE_pct"])
    ap = float(r["patch_AP_pct"])
    ax.scatter(x, y, s=55, c=color, marker=marker, zorder=5)
    ax.annotate(
        f"{name}\nAP {ap:.2f}%",
        (x, y),
        xytext=position,
        textcoords="data",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": color, "lw": 0.6},
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
    )
ax.set(
    xlim=(19, 41),
    ylim=(63.15, 68.15),
    xlabel="Object contrast capture (%)",
    ylabel="Reconstruction VE (%)",
)
ax.grid(alpha=0.18, lw=0.5)
ax.set_xticks([20, 30, 40])
ax.set_yticks([64, 65, 66, 67])
save(fig, "agf_width_balance")

js = json.loads((R / "agf_loss_ablation_main.json").read_text())
vals = np.array(js["percentages"])
sd = np.array(js["sample_SD"])
fig, axs = plt.subplots(1, 2, figsize=(2.95, 1.58), layout="constrained")
for j, ax in enumerate(axs):
    for i in range(4):
        ax.plot([vals[0, j], vals[i, j]], [i, i], color=colors[i], lw=1.6)
        ax.errorbar(
            vals[i, j],
            i,
            xerr=sd[i, j],
            fmt="o",
            markersize=4,
            color=colors[i],
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
        yticks=np.arange(4),
        yticklabels=["Recon.", "+ contrast", "+ detection", "+ both"] if j == 0 else [],
        xlim=(18, 40) if j == 0 else (57, 77),
        xlabel=["Capture (%)", "Patch AP (%)"][j],
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([20, 30] if j == 0 else [60, 70])
    ax.spines["left"].set_visible(False)
save(fig, "agf_loss_lollipop")

fig, ax = plt.subplots(figsize=(3.0, 1.80), layout="constrained")
x = np.arange(3)
vals = np.array([[15.88, 50.71, 23.77], [25.48, 69.07, 51.92]])
for j, (name, color) in enumerate([("SLiCS", gray), ("GGF", teal)]):
    bars = ax.bar(x + (j - 0.5) * 0.34, vals[j], width=0.32, label=name, color=color)
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in vals[j]], padding=3, fontsize=10)
ax.set(
    xticks=x, xticklabels=["Capture", "Patch AP", "VE"], ylim=(0, 92), ylabel="Percent"
)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    ncol=2,
    fontsize=9,
    handlelength=1.1,
    columnspacing=0.8,
)
ax.set_yticks([0, 40, 80])
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
save(fig, "agf_slics_bars")

js = json.loads((R / "agf_target_capture_bars.json").read_text())
fig, ax = plt.subplots(figsize=(3.0, 1.94), layout="constrained")
for i, ((seed, cat), (a, b)) in enumerate(
    zip(js["targets"], js["capture_percentages"])
):
    ax.plot([a, b], [i, i], color=gray, lw=2)
    ax.scatter(
        a, i, color=purple, s=27, marker="o", label="GGF" if i == 0 else None, zorder=4
    )
    ax.scatter(
        b, i, color=teal, s=30, marker="D", label="GGF+X" if i == 0 else None, zorder=5
    )
    ax.text(b + 0.25, i, f"+{b - a:.2f}", fontsize=10, va="center")
ax.set(
    yticks=np.arange(5),
    yticklabels=[f"{cat.title()} ({seed})" for seed, cat in js["targets"]],
    xlim=(10, 37),
    xlabel="Capture (%)",
)
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.legend(frameon=False, ncol=2, loc="lower right", bbox_to_anchor=(1, 1.02))
save(fig, "agf_exchange_dumbbell")

fig, ax = plt.subplots(figsize=(3.0, 1.80), layout="constrained")
for seed, color in zip([21, 22, 23], [teal, purple, amber]):
    pts = []
    for phase, offset in [(1, 0), (2, 2500)]:
        hist = json.loads(
            (
                B
                / f"evidence/width_balance/histories/phase{phase}_mask4_balanced_s{seed}.json"
            ).read_text()
        )
        for row in hist:
            w = row.get("weights_next") or row.get("weights_used")
            if w:
                pts.append((offset + row["update"], w[1], w[2]))
    pts.sort()
    a = np.array(pts)
    for j, style in [(1, "-"), (2, "--")]:
        ax.plot(
            a[:, 0],
            a[:, j],
            color=color,
            ls=style,
            lw=1.1,
            label=f"{seed}" if j == 1 else None,
        )
    print("Final weights", seed, pts[-1])
ax.axvline(2500, color=gray, ls=":", lw=0.7)
ax.set(xlabel="Guided updates", ylabel="Loss weight", ylim=(0, 0.23))
ax.legend(title="Seed", ncol=3, frameon=False, loc="lower right")
ax.grid(alpha=0.15)
save(fig, "agf_weight_history")


# Regenerate the schematic separately with make_architecture.py.

# Completed guided continuation histories; preserve measured checkpoint values.
fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.15), layout="constrained")
for seed, color in zip([21, 22, 23], [teal, purple, amber]):
    for arm, style in [("continue", "-"), ("pcgrad", "--")]:
        hist = json.loads(
            (
                B
                / f"evidence/continuation/histories/continuation_ggf_{arm}_s{seed}.json"
            ).read_text()
        )
        if isinstance(hist, dict):
            hist = hist["history"]
        for ax, key, title in zip(
            axs,
            ["macro_capture", "macro_ap", "ve"],
            ["Capture (%)", "Patch AP (%)", "Reconstruction VE (%)"],
        ):
            vals = [
                100
                * (
                    row["validation"]["reconstruction"]["ve"]
                    if key == "ve"
                    else row["validation"][key]
                )
                for row in hist
            ]
            ax.plot(
                [row["update"] for row in hist], vals, color=color, ls=style, lw=1.05
            )
            ax.set(xlabel="Additional updates", ylabel=title)
            ax.set_xticks([0, 1500, 3000])
            ax.grid(alpha=0.15)
from matplotlib.lines import Line2D

handles = [
    Line2D([0], [0], color=c, label=str(s))
    for s, c in zip([21, 22, 23], [teal, purple, amber])
]
handles += [
    Line2D([0], [0], color=gray, ls="-", label="Ordinary"),
    Line2D([0], [0], color=gray, ls="--", label="PCGrad"),
]
fig.legend(
    handles=handles, ncol=5, frameon=False, loc="outside upper center", fontsize=10
)
save(fig, "agf_continuation_curves")

# Completed current ordinary 5D continuation; development only.
fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.0), layout="constrained")
for seed, color in zip([21, 22, 23], [teal, purple, amber]):
    hist = json.loads(
        (
            B / f"evidence/continuation5/histories/guided5_continue_s{seed}.json"
        ).read_text()
    )
    if isinstance(hist, dict):
        hist = hist["history"]
    for ax, key, title in zip(
        axs,
        ["macro_capture", "macro_ap", "ve"],
        ["Capture (%)", "Patch AP (%)", "Reconstruction VE (%)"],
    ):
        vals = [
            100
            * (
                row["validation"]["reconstruction"]["ve"]
                if key == "ve"
                else row["validation"][key]
            )
            for row in hist
        ]
        ax.plot(
            [row["update"] for row in hist], vals, color=color, lw=1.3, label=str(seed)
        )
        ax.set(xlabel="Additional updates", ylabel=title)
        ax.set_xticks([0, 1500, 3000])
        ax.grid(alpha=0.15)
fig.legend(title="Seed", frameon=False, ncol=3, loc="outside upper center", fontsize=10)
save(fig, "agf_continuation5_curves")

# Keep current compact layouts as the final rendering pass.
runpy.run_path(str(B / "figure_sources/make_compact_figures.py"), run_name="__main__")
