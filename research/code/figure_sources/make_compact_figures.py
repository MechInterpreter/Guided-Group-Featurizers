"""Compact scientific plots from saved records; no model training or evaluation."""

from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pymupdf

R = Path(__file__).resolve().parent.parent
I = R / "figure_sources/inputs"
O = R / "figures"
O.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.65,
    }
)
TEAL = "#007D75"
PURPLE = "#7951A0"
BLUE = "#527E9D"
AMBER = "#AD761E"
GRAY = "#77868C"


def save(fig, name):
    fig.savefig(O / (name + ".pdf"))
    fig.savefig(O / (name + ".svg"))
    fig.savefig(O / (name + ".png"), dpi=180)
    plt.close(fig)


vals = np.array(
    json.loads((I / "agf_loss_ablation_main.json").read_text())["percentages"]
)
sd = np.array(json.loads((I / "agf_loss_ablation_main.json").read_text())["sample_SD"])
fig, axs = plt.subplots(1, 2, figsize=(2.95, 1.40), layout="constrained")
for j, ax in enumerate(axs):
    for i, color in enumerate([BLUE, GRAY, AMBER, PURPLE]):
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
save(fig, "agf_loss_lollipop")

fig, ax = plt.subplots(figsize=(3.0, 1.60), layout="constrained")
x = np.arange(3)
# Exact displayed means from the unchanged matched-supervised comparison.
vals = np.array([[15.88, 50.71, 23.77], [25.48, 69.07, 51.92]])
for j, (name, color) in enumerate([("SLiCS", GRAY), ("GGF", TEAL)]):
    bars = ax.bar(x + (j - 0.5) * 0.48, vals[j], width=0.32, label=name, color=color)
    labels = ax.bar_label(
        bars, labels=[f"{v:.2f}" for v in vals[j]], padding=3, fontsize=10
    )
    # Stagger the close Capture labels without shrinking text or changing data.
    if j == 1:
        labels[0].set_position((0, 10))
ax.set(
    xticks=x,
    xticklabels=["Capture", "Patch AP", "VE"],
    ylim=(0, 92),
    ylabel="Percent",
    yticks=[0, 40, 80],
)
ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.0),
    frameon=False,
    ncol=2,
    fontsize=10,
    handlelength=0.9,
    columnspacing=0.7,
)
ax.grid(axis="y", alpha=0.15)
ax.set_axisbelow(True)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
value_labels = [
    a for a in ax.texts if a.get_text() in {f"{v:.2f}" for row in vals for v in row}
]
assert len(value_labels) == 6
for i, a in enumerate(value_labels):
    for b in value_labels[i + 1 :]:
        assert (
            not a.get_window_extent(renderer)
            .expanded(1.04, 1.10)
            .overlaps(b.get_window_extent(renderer))
        ), (a.get_text(), b.get_text())
    for bar in ax.patches:
        assert not a.get_window_extent(renderer).overlaps(
            bar.get_window_extent(renderer)
        ), ("Label overlaps a bar", a.get_text())
save(fig, "agf_slics_bars")

fig, ax = plt.subplots(figsize=(3.0, 1.52), layout="constrained")
for seed, color, marker in zip([21, 22, 23], [TEAL, PURPLE, AMBER], ["o", "s", "^"]):
    pts = []
    for phase, offset in [(1, 0), (2, 2500)]:
        hist = json.loads(
            (
                R
                / f"evidence/width_balance/histories/phase{phase}_mask4_balanced_s{seed}.json"
            ).read_text()
        )
        for row in hist:
            w = row.get("weights_next") or row.get("weights_used")
            if w:
                pts.append((offset + row["update"], w[1], w[2]))
    a = np.array(sorted(pts))
    for j, style in [(1, "-"), (2, "--")]:
        ax.plot(
            a[:, 0],
            a[:, j],
            color=color,
            ls=style,
            lw=1.05,
            marker=marker,
            markersize=2,
            markevery=max(1, len(a) // 4),
            label=str(seed) if j == 1 else None,
        )
ax.axvline(2500, color=GRAY, ls=":", lw=0.7)
ax.set(xlabel="Guided updates", ylabel="Loss weight", ylim=(0, 0.23))
ax.legend(ncol=3, frameon=False, loc="lower right", handlelength=1.1, columnspacing=0.7)
ax.grid(alpha=0.12)
save(fig, "agf_weight_history")

rows = list(csv.DictReader((R / "evidence/width_balance/three_seed_means.csv").open()))
fig, ax = plt.subplots(figsize=(3.0, 1.70), layout="constrained")
for row, name, col, marker, pos in zip(
    rows,
    ["3D fixed", "4D fixed", "3D adaptive", "4D adaptive", "Vanilla 4D"],
    [PURPLE, BLUE, AMBER, TEAL, GRAY],
    ["o", "s", "^", "D", "X"],
    [(25, 65.25), (28, 67.65), (26.5, 63.5), (34, 66.5), (19.8, 67.5)],
):
    x = float(row["capture_pct"])
    y = float(row["VE_pct"])
    ap = float(row["patch_AP_pct"])
    ax.scatter(x, y, s=40, c=col, marker=marker, zorder=5)
    ax.annotate(
        f"{name}\nAP {ap:.2f}%",
        (x, y),
        xytext=pos,
        textcoords="data",
        fontsize=9.6,
        arrowprops={"arrowstyle": "-", "color": col, "lw": 0.6},
        color=col,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.1},
    )
ax.set(
    xlim=(19, 41),
    ylim=(63.15, 68.15),
    xlabel="Capture (%)",
    ylabel="VE (%)",
    xticks=[20, 30, 40],
    yticks=[64, 66, 68],
)
ax.grid(alpha=0.15, lw=0.5)
save(fig, "agf_width_balance")

pilot = list(csv.DictReader((R / "evidence/oct15/priority_pilot_means.csv").open()))
fig, ax = plt.subplots(figsize=(3.0, 1.45), layout="constrained")
x = np.arange(3)
for j, (row, col, name) in enumerate(zip(pilot, [TEAL, GRAY], ["GGF", "Vanilla"])):
    v = [float(row[k]) for k in ["capture_pct", "patch_AP_pct", "VE_pct"]]
    bars = ax.bar(x + (j - 0.5) * 0.34, v, width=0.32, color=col, label=name)
    ax.bar_label(
        bars, labels=[f"{a:.2f}" for a in v], padding=13 if j == 0 else 1, fontsize=10
    )
ax.set(
    xticks=x,
    xticklabels=["Capture", "Patch AP", "VE"],
    ylim=(0, 120),
    yticks=[0, 40, 80],
)
ax.legend(
    frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.27), handlelength=1
)
save(fig, "agf_pilot")

fig, axs = plt.subplots(1, 3, figsize=(6.875, 1.65), layout="constrained")
for seed, col, marker in zip([21, 22, 23], [TEAL, PURPLE, AMBER], ["o", "s", "^"]):
    hist = json.loads(
        (
            R / f"evidence/continuation5/histories/guided5_continue_s{seed}.json"
        ).read_text()
    )
    if isinstance(hist, dict):
        hist = hist["history"]
    for ax, key, title in zip(
        axs,
        ["macro_capture", "macro_ap", "ve"],
        ["Capture (%)", "Patch AP (%)", "VE (%)"],
    ):
        v = [
            100
            * (
                q["validation"]["reconstruction"]["ve"]
                if key == "ve"
                else q["validation"][key]
            )
            for q in hist
        ]
        ax.plot(
            [q["update"] for q in hist],
            v,
            color=col,
            lw=1,
            marker=marker,
            markersize=2.5,
            markevery=max(1, len(hist) // 5),
            label=str(seed),
        )
        ax.set(xlabel="Additional updates", ylabel=title, xticks=[0, 1500, 3000])
        ax.grid(alpha=0.15)
fig.legend(
    *axs[0].get_legend_handles_labels(),
    title="Seed",
    frameon=False,
    ncol=3,
    loc="outside upper center",
    handlelength=1.4,
    columnspacing=1.1,
)
save(fig, "agf_continuation5_curves")

# All 75 current categories in one compact, fully labeled figure.
source = R / "evidence/pilot4096_test/test_categories.csv"
rows = list(csv.DictReader(source.open()))
assert len(rows) == 450 and {int(q["seed"]) for q in rows} == {21, 22, 23}
names = {int(q["category_index"]): q["category"] for q in rows}
inds = sorted(names)
assert len(inds) == 75
means = []
for ci in inds:
    entry = {"category_index": ci, "category": names[ci]}
    for kind, prefix in [("released_vanilla", "V"), ("agf_g", "G")]:
        q = [r for r in rows if int(r["category_index"]) == ci and r["kind"] == kind]
        assert len(q) == 3 and {int(r["seed"]) for r in q} == {21, 22, 23}
        for metric in ["capture", "ap"]:
            entry[f"{prefix}_{metric}_pct"] = 100 * np.mean(
                [float(r[metric]) for r in q]
            )
    means.append(entry)
assert np.isclose(np.mean([q["G_capture_pct"] for q in means]), 42.77, atol=0.005)
assert np.isclose(np.mean([q["G_ap_pct"] for q in means]), 74.76, atol=0.005)
out = R / "evidence/pilot4096_test/current_category_means.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(means[0]))
    w.writeheader()
    w.writerows(means)
fig = plt.figure(figsize=(6.875, 4.83), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 495), ylim=(347.76, 0))
ax.axis("off")
for panel in range(3):
    base = panel * 165
    for x, s in [(base + 88, "Capture"), (base + 138, "Patch AP")]:
        ax.text(x, 10, s, fontsize=9.2, ha="center", va="center")
    for j, s in enumerate(["V", "G", "V", "G"]):
        ax.text(base + 75.5 + j * 25, 25, s, fontsize=9.2, ha="center", va="center")
    for i, row in enumerate(means[panel * 25 : (panel + 1) * 25]):
        y = 37 + i * 11.5
        ax.text(
            base + 61,
            y,
            row["category"].replace("_", " "),
            fontsize=9.2,
            fontfamily="Arial",
            ha="right",
            va="center",
        )
        for j, key in enumerate(
            ["V_capture_pct", "G_capture_pct", "V_ap_pct", "G_ap_pct"]
        ):
            v = row[key]
            x = base + 63 + j * 25
            ax.add_patch(
                Rectangle(
                    (x, y - 5.5),
                    24.6,
                    11.1,
                    facecolor=plt.get_cmap("Blues")(0.08 + 0.85 * v / 100),
                    edgecolor="none",
                )
            )
            ax.text(
                x + 12.5,
                y,
                f"{v:.2f}",
                fontsize=9.2,
                fontfamily="Arial",
                ha="center",
                va="center",
                color="white" if v >= 65 else "#17252B",
            )
# Shared scale and its numeric ticks, without consuming a plot panel.
for j in range(100):
    ax.add_patch(
        Rectangle(
            (163 + j, 329),
            1.1,
            4.5,
            facecolor=plt.get_cmap("Blues")(0.08 + 0.85 * j / 99),
            edgecolor="none",
        )
    )
for v in [0, 25, 50, 75, 100]:
    ax.text(163 + v, 342, str(v), fontsize=9.2, ha="center", va="center")
ax.text(145, 331, "%", fontsize=9.2, ha="right", va="center")
save(fig, "ggf_current_categories")
(I / "current_category_figure_audit.json").write_text(
    json.dumps(
        {
            "source": "evidence/pilot4096_test/test_categories.csv",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "seeds": [21, 22, 23],
            "groups": 4096,
            "directions_per_group": 12,
            "top_k": 32,
            "cap": 0.6,
            "category_count": 75,
            "averaging": "arithmetic mean over seeds within each category",
            "retained_all_categories": True,
            "font_size_pt": 9.2,
        },
        indent=2,
    )
)
print("Rendered compact figures and current 75-category comparison from saved records.")


# Narrow vector coverage chart: retain all printed source summary values.
record = json.loads((I / "coverage_display_records.json").read_text())
fig = plt.figure(figsize=(3.25, 1.40), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 234), ylim=(100.8, 0))
ax.axis("off")
for value in [10, 100, 1000]:
    px = 70 + 50 * np.log10(value / 10)
    ax.plot([px, px], [6, 73], color="#E7ECED", lw=0.65, zorder=0)
    ax.text(px, 82, f"{value}", ha="center", va="center", fontsize=9.5)
for j, (name, lo, med, hi) in enumerate(record["rows"]):
    y = 12 + j * 18
    ax.text(62, y, name, ha="right", va="center", fontsize=9.5)
    x = [70 + 50 * np.log10(a / 10) for a in [lo, med, hi]]
    ax.plot([x[0], x[2]], [y, y], color="#ADBDC5", lw=2.4)
    ax.plot(x[1], y, "o", color=TEAL, ms=4)
    ax.text(233, y, f"{lo}/{med}/{hi}", ha="right", va="center", fontsize=9.5)
ax.text(
    125, 94, "Usable images per category (log)", ha="center", va="center", fontsize=9.5
)
save(fig, "agf_plot_coverage")
