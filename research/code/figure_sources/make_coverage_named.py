"""COCO split coverage with protocol names; saved numeric records unchanged."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parents[1]
O = R / "figures"
O.mkdir(exist_ok=True)
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
record = json.loads(
    (R / "figure_sources/inputs/coverage_display_records.json").read_text()
)
fig = plt.figure(figsize=(3.25, 1.40), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 234), ylim=(100.8, 0))
ax.axis("off")
for value in [10, 100, 1000]:
    px = 70 + 50 * np.log10(value / 10)
    ax.plot([px, px], [6, 73], color="#E7ECED", lw=0.65, zorder=0)
    ax.text(px, 82, f"{value}", ha="center", va="center", fontsize=9.5)
for j, (name, lo, med, hi) in enumerate(record["rows"]):
    name = "Test (800)" if name == "Earlier test" else name
    y = 12 + j * 18
    ax.text(62, y, name, ha="right", va="center", fontsize=9.5)
    x = [70 + 50 * np.log10(a / 10) for a in [lo, med, hi]]
    ax.plot([x[0], x[2]], [y, y], color="#ADBDC5", lw=2.4)
    ax.plot(x[1], y, "o", color="#007D75", ms=4)
    ax.text(233, y, f"{lo}/{med}/{hi}", ha="right", va="center", fontsize=9.5)
ax.text(
    125, 94, "Usable images per category (log)", ha="center", va="center", fontsize=9.5
)
fig.savefig(
    O / "agf_plot_coverage_named.pdf", metadata={"CreationDate": None, "ModDate": None}
)
fig.savefig(O / "agf_plot_coverage_named.png", dpi=220)
plt.close(fig)
