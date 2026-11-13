"""Current component ablation from verified saved three-seed test records."""

from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

R = Path(__file__).resolve().parent.parent
(R / "figures").mkdir(exist_ok=True)
source = R / "evidence/components_current/test_means.csv"
rows = list(csv.DictReader(source.open()))
order = ["released_vanilla", "mask_only", "fixed_guided", "agf_g"]
rows = [next(r for r in rows if r["kind"] == k) for k in order]
colors = ["#768B9A", "#D19B46", "#71AEA3", "#007F78"]
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
fig = plt.figure(figsize=(3.25, 1.90), facecolor="white")
axs = [fig.add_axes([0.255 + j * 0.251, 0.21, 0.213, 0.68]) for j in range(3)]
for ax, key, title, lim, ticks in zip(
    axs,
    ["capture_pct", "patch_AP_pct", "VE_pct"],
    ["Capture", "Patch AP", "VE"],
    [(0, 50), (0, 100), (74, 82)],
    [[0, 40], [0, 80], [74, 82]],
):
    for j, r in enumerate(rows):
        v = float(r[key])
        sd = float(r[key + "_seed_sd"])
        ax.errorbar(
            v,
            j,
            xerr=sd,
            fmt="o",
            ms=4.3,
            color=colors[j],
            elinewidth=1,
            capsize=2,
            capthick=0.8,
            zorder=4,
        )
        ax.text(
            0.99,
            j - 0.22,
            f"{v:.2f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=9,
            color="#263B44",
        )
    ax.set(
        xlim=lim,
        ylim=(3.48, -0.74),
        xticks=ticks,
        yticks=range(4),
        yticklabels=["Vanilla", "Mask init.", "Fixed", "GGF"] if ax is axs[0] else [],
    )
    ax.set_title(title, pad=5, fontsize=9, weight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=2, pad=3)
    ax.grid(axis="x", color="#DFE5E8", lw=0.5, zorder=0)
    for y in [0.5, 1.5, 2.5]:
        ax.axhline(y, color="#EDF0F2", lw=0.5, zorder=0)
fig.savefig(R / "figures/ggf_components.pdf")
fig.savefig(R / "figures/ggf_components.svg")
fig.savefig(R / "figures/ggf_components.png", dpi=240)
plt.close(fig)
with pymupdf.open(R / "figures/ggf_components.pdf") as d:
    sizes = []
    for block in d[0].get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                assert (d[0].rect + (-0.1, -0.1, 0.1, 0.1)).contains(
                    pymupdf.Rect(span["bbox"])
                ), span["text"]
                sizes.append(span["size"])
audit = dict(
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    rows=order,
    values="Unrounded saved three-seed means; whiskers are sample SD.",
    minimum_final_font_pt=min(sizes),
    placement_width_bp=234,
    VE_axis=[74, 82],
    bar_chart=False,
)
(R / "figure_sources/inputs/components_plot.json").write_text(
    json.dumps(audit, indent=2)
)
print(json.dumps(audit, indent=2))
