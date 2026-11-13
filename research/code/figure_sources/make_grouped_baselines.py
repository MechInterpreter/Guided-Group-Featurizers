"""Group displayed baseline rows by model family; preserve measured values."""

from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pymupdf

R = Path(__file__).resolve().parent.parent
(R / "figures").mkdir(exist_ok=True)
source = R / "evidence/baseline_comparison.csv"
rows = list(csv.DictReader(source.open()))
order = [0, 4, 6, 8, 1, 2, 3, 5, 7, 9]
labels = [
    "Vanilla BSF 3",
    "Vanilla 8 cont.",
    "Vanilla 12 exp.",
    "Vanilla 12 fresh",
    "Grassmannian 3",
    "Group Lasso 3",
    "GGF 3",
    "GGF 8 cont.",
    "GGF 12 exp.",
    "GGF 12 fresh",
]
colors = [
    "#A3B9CC",
    "#7395B4",
    "#507695",
    "#2D506E",
    "#859096",
    "#B08643",
    "#A8CBC4",
    "#69A69C",
    "#287F75",
    "#005B55",
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "savefig.facecolor": "white",
    }
)
selected = [rows[i] for i in order]
fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.10), layout="constrained")
for ax, key, title in zip(
    axs,
    ["capture_pct", "patch_AP_pct", "VE_pct"],
    ["Capture (%)", "Patch AP (%)", "Reconstruction VE (%)"],
):
    y = np.arange(len(selected))
    values = [float(r[key]) for r in selected]
    ax.barh(y, values, color=colors, height=0.64)
    for yi, value in zip(y, values):
        ax.text(value + 0.8, yi, f"{value:.2f}", va="center", fontsize=10.5)
    ax.set(
        yticks=y,
        yticklabels=labels if ax is axs[0] else [],
        xlim=(0, 54 if key == "capture_pct" else 102),
        xlabel=title,
    )
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0, 20, 40] if key == "capture_pct" else [0, 40, 80])
    for cut in [3.5, 5.5]:
        ax.axhline(cut, color="#B5BFC3", lw=0.6)
fig.savefig(R / "figures/agf_released_bars.pdf")
fig.savefig(R / "figures/agf_released_bars.svg")
fig.savefig(R / "agf_released_bars.png", dpi=180)
plt.close(fig)
with pymupdf.open(R / "figures/agf_released_bars.pdf") as d:
    for b in d[0].get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for span in line["spans"]:
                assert (d[0].rect + (-0.5, -0.5, 0.5, 0.5)).contains(
                    pymupdf.Rect(span["bbox"])
                ), span["text"]
(R / "figure_sources/inputs/baseline_display_order.json").write_text(
    json.dumps(
        dict(
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_row_indices=order,
            models=[r["model"] for r in selected],
            grouping=["Vanilla", "Other released baselines", "GGF"],
            ordering="Original 3D, 8D continued, 12D expanded, 12D fresh within each family; identical ordering in all metrics.",
            values_unchanged=True,
        ),
        indent=2,
    )
)
print("Grouped baseline rows; all measured values unchanged.")
