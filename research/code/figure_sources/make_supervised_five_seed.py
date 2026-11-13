"""Five-order completed-training results; no training/evaluation."""

from pathlib import Path
import csv, hashlib, json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

R = Path(__file__).resolve().parent.parent
E = R / "evidence/supervised_five_seed"
O = R / "figures"
O.mkdir(exist_ok=True)
records = list(csv.DictReader((E / "supervised_means.csv").open()))
arms = ["matched_ggf_final", "slics_final_partial"]
rows = [
    next(r for r in records if r["arm"] == a and r["seed_group"] == "all_five")
    for a in arms
]
colors = ["#007F78", "#77868C"]
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
        "axes.spines.left": False,
        "axes.linewidth": 0.5,
    }
)
fig = plt.figure(figsize=(6, 1.45), facecolor="white")
axes = [fig.add_axes([0.225 + i * 0.254, 0.32, 0.218, 0.50]) for i in range(3)]
for ax, key, title in zip(
    axes,
    ["capture_pct", "patch_AP_pct", "VE_pct"],
    ["Capture", "Patch AP", "Reconstruction VE"],
):
    for y, (r, color) in enumerate(zip(rows, colors)):
        mean = float(r[key])
        lo = float(r[key + "_ci_low"])
        hi = float(r[key + "_ci_high"])
        ax.errorbar(
            mean,
            y,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            ms=4.4,
            color=color,
            capsize=2,
            elinewidth=1,
            capthick=0.8,
            zorder=4,
        )
        ax.text(
            mean,
            y - 0.22,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#22343C",
        )
    ax.set(
        xlim=(0, 100),
        ylim=(1.45, -0.72),
        xticks=[0, 50, 100],
        yticks=range(2),
        yticklabels=["GGF", "SLiCS"] if ax is axes[0] else [],
    )
    ax.set_title(title, fontsize=9, weight="bold", pad=7)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.tick_params(axis="x", length=2, pad=3)
    ax.grid(axis="x", color="#E0E6E8", linewidth=0.5, zorder=0)
    ax.set_xlabel("Percent", labelpad=3)
for ext in ["pdf", "svg"]:
    fig.savefig(O / ("ggf_supervised_five_seed." + ext))
fig.savefig(O / "ggf_supervised_five_seed.png", dpi=240)
plt.close(fig)

# Regenerate only the earlier SLiCS chart, retaining all six displayed values.
fig, ax = plt.subplots(figsize=(3.0, 1.60), layout="constrained")
x = np.arange(3)
vals = np.array([[15.88, 50.71, 23.77], [25.48, 69.07, 51.92]])
for j, (name, color) in enumerate([("SLiCS", colors[1]), ("GGF", colors[0])]):
    bars = ax.bar(x + (j - 0.5) * 0.48, vals[j], width=0.32, label=name, color=color)
    labels = ax.bar_label(
        bars, labels=[f"{v:.2f}" for v in vals[j]], padding=3, fontsize=10
    )
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
for ext in ["pdf", "svg"]:
    fig.savefig(O / ("agf_slics_bars." + ext))
fig.savefig(O / "agf_slics_bars.png", dpi=200)
plt.close(fig)

with pymupdf.open(O / "ggf_supervised_five_seed.pdf") as d:
    spans = [
        s
        for b in d[0].get_text("dict")["blocks"]
        for l in b.get("lines", [])
        for s in l["spans"]
    ]
    for s in spans:
        assert (d[0].rect + (-0.2, -0.2, 0.2, 0.2)).contains(pymupdf.Rect(s["bbox"])), (
            s["text"]
        )
    minimum = min(s["size"] for s in spans)
    assert minimum >= 8.99
audit = dict(
    source_sha256=hashlib.sha256((E / "supervised_means.csv").read_bytes()).hexdigest(),
    arms=arms,
    seed_group="all_five",
    whiskers="95% Student-t intervals over five training orders; not image-bootstrap bands",
    comparison="Completed-training GGF versus completed-training SLiCS only; selected checkpoint retained in appendix",
    allocated_figure_width_bp=432,
    minimum_font_pt=minimum,
    all_axes=[0, 100],
    values=[
        {k: r[k] for k in ["arm", "capture_pct", "patch_AP_pct", "VE_pct"]}
        for r in rows
    ],
)
(E / "figure_audit.json").write_text(json.dumps(audit, indent=2))
print(json.dumps(audit, indent=2))
