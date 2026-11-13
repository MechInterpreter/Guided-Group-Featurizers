"""Rebuild the complete capacity overview from audited display inputs.

Run from any directory: python figure_sources/make_capacity_overview_named.py
Source means and sample SD are recorded in inputs/capacity_overview.
The original audit checks all 14 paired rows against per-run CSV evidence.
"""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).resolve().parent
data = json.loads((H / "inputs/capacity_overview/capacity_s7_data.json").read_text())
rows, headers = data["rows"], data["headers"]
y = max(r["y"] for r in rows) + 1
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.2,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)
fig, axes = plt.subplots(
    1,
    3,
    figsize=(6.875, 7.45),
    gridspec_kw={
        "left": 0.416,
        "right": 0.99,
        "top": 0.957,
        "bottom": 0.052,
        "wspace": 0.21,
    },
)
colors = {
    "agf_g": "#007f79",
    "released_vanilla": "#84949a",
    "supervised_vanilla": "#45a7b5",
    "historical": "#9c92b6",
}
for j, (ax, title) in enumerate(
    zip(axes, ["Capture (%)", "Patch AP (%)", "Recon. VE (%)"])
):
    for r in rows:
        ax.barh(
            r["y"],
            r["means"][j],
            height=0.61,
            color=colors[r["kind"]],
            xerr=r["sds"][j],
            error_kw=dict(ecolor="#26343a", elinewidth=0.65, capsize=1.6, capthick=0.6),
        )
        ax.text(
            r["means"][j] + r["sds"][j] + 1,
            r["y"],
            f"{r['means'][j]:.2f}",
            va="center",
            fontsize=6.6,
        )
    ax.set_xlim(0, 103)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(y + 0.1, -1.1)
    ax.set_title(title, fontsize=8.4, pad=7)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["label"] for r in rows] if j == 0 else [], fontsize=7.6)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.tick_params(axis="x", length=2, labelsize=7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b4bfc3")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8ecee", lw=0.5)
    for yy, text in headers:
        text = text.replace("Historical studies:", "Capacity studies:")
        if j == 0:
            fig.text(
                0.03,
                fig.transFigure.inverted().transform(ax.transData.transform((0, yy)))[
                    1
                ],
                text,
                fontweight="bold",
                fontsize=7.9,
                va="center",
            )
        if yy > 0:
            ax.axhline(yy - 0.6, color="#cbd4d7", lw=0.6, xmin=0, xmax=1)
out = H.parent / "figures"
out.mkdir(exist_ok=True)
for ext in ["pdf", "svg", "png"]:
    fig.savefig(out / f"ggf_capacity_comparison_overview_named.{ext}", dpi=190)
plt.close(fig)
