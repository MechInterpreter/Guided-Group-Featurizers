"""Combined interpretability figure from the archived seven run-level scores."""

from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "evidence/matched75/RESULTS_SEEDS21_27.json").read_text())
assert DATA["complete"] and DATA["seeds"] == [21, 22, 23, 24, 25, 26, 27]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
NAMES = {
    "ggf": "GGF",
    "slics": "SLiCS",
    "supervised_vanilla": "Supervised Vanilla",
    "vanilla": "Vanilla",
}
COLORS = {
    "ggf": "#09715d",
    "slics": "#536fa7",
    "supervised_vanilla": "#a16c35",
    "vanilla": "#7b718a",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
CONFIGS = [
    (
        "full_category",
        ["ggf", "slics", "supervised_vanilla", "vanilla"],
        "All category labels",
    ),
    (
        "category_split",
        ["ggf", "supervised_vanilla", "vanilla"],
        "Category-split guidance",
    ),
]
fig = plt.figure(figsize=(7.0, 4.05))
top = [
    fig.add_axes([0.135, 0.570, 0.330, 0.360]),
    fig.add_axes([0.650, 0.570, 0.335, 0.360]),
]
bottom = [
    fig.add_axes([0.080, 0.140, 0.400, 0.220]),
    fig.add_axes([0.585, 0.140, 0.400, 0.220]),
]
labels = []
for i, (ax, (regime, methods, title)) in enumerate(zip(top, CONFIGS)):
    yy = np.arange(len(methods))
    for offset, metric, label, color in [
        (-0.23, "semantic", "Specific pairs", "#09715d"),
        (0, "visual", "Visual pairs", "#536fa7"),
        (0.23, "clear_rate", "Whole groups", "#9c733e"),
    ]:
        values = [
            next(
                r
                for r in DATA["coherence"]
                if (r["regime"], r["method"]) == (regime, m)
            )["metrics"][metric]
            for m in methods
        ]
        ax.barh(
            yy + offset,
            [r["mean"] * 100 for r in values],
            xerr=[r["sd"] * 100 for r in values],
            height=0.20,
            color=color,
            label=label,
            error_kw={"elinewidth": 0.65, "capsize": 1.4, "capthick": 0.65},
            zorder=3,
        )
        for y, m, value in zip(yy + offset, methods, values):
            text = f"{value['mean'] * 100:.1f}%"
            ax.text(
                (value["mean"] + value["sd"]) * 100 + 1.0,
                y,
                text,
                va="center",
                fontsize=7.0,
                clip_on=False,
            )
            labels.append(
                dict(
                    regime=regime,
                    method=m,
                    endpoint=metric,
                    text=text,
                    mean=value["mean"],
                    sd=value["sd"],
                )
            )
    ax.set_yticks(
        yy,
        [
            NAMES[m].replace("Supervised Vanilla", "Supervised\nVanilla")
            for m in methods
        ],
        fontsize=7.8,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 120)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("Confirmed pairs / clear groups (%)", fontsize=7.5)
    ax.set_title(
        f"({chr(97 + i)}) {title}", loc="left", fontsize=8.5, fontweight="bold", pad=8
    )
    ax.grid(axis="x", color="#e4e7e9", linewidth=0.5, zorder=0)
handles, names = top[0].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.55, 0.440),
    ncol=3,
    frameon=False,
    fontsize=7.5,
)
for i, (ax, (regime, methods, _)) in enumerate(zip(bottom, CONFIGS)):
    stratum = "all_categories" if regime == "full_category" else "excluded"
    for m in methods:
        rr = [
            next(
                r
                for r in DATA["concentration"]
                if (
                    r["regime"],
                    r["method"],
                    r["stratum"],
                    r["negative_quantile"],
                    r["budget"],
                )
                == (regime, m, stratum, 0.95, k)
            )
            for k in [1, 2, 4, 8]
        ]
        x = [r["metrics"]["OR_FPR"]["mean"] * 100 for r in rr]
        y = [r["metrics"]["OR_coverage"]["mean"] * 100 for r in rr]
        ax.errorbar(
            x,
            y,
            xerr=[r["metrics"]["OR_FPR"]["sd"] * 100 for r in rr],
            yerr=[r["metrics"]["OR_coverage"]["sd"] * 100 for r in rr],
            fmt="-o",
            color=COLORS[m],
            markersize=3.3,
            linewidth=1.15,
            elinewidth=0.65,
            capsize=1.5,
            label=NAMES[m],
            zorder=3,
        )
        if m == "ggf":
            for a, b, k in zip(x, y, [1, 2, 4, 8]):
                ax.annotate(
                    str(k),
                    (a, b),
                    xytext=(2, 4),
                    textcoords="offset points",
                    fontsize=6.8,
                    color=COLORS[m],
                )
    ax.set_xlim(0, 36)
    ax.set_ylim(40, 104)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([40, 60, 80, 100])
    ax.tick_params(labelsize=7)
    ax.grid(color="#e4e7e9", linewidth=0.5)
    ax.set_xlabel("False positives (%)", fontsize=7.8)
    ax.set_ylabel("Target coverage (%)", fontsize=7.8)
    heading = (
        "All category labels"
        if regime == "full_category"
        else "Categories withheld from guidance"
    )
    ax.set_title(
        f"({chr(99 + i)}) {heading}", loc="left", fontsize=8.2, fontweight="bold", pad=8
    )
handles, names = bottom[0].get_legend_handles_labels()
fig.legend(
    handles,
    names,
    loc="center",
    bbox_to_anchor=(0.55, 0.015),
    ncol=4,
    frameon=False,
    fontsize=7.2,
)
fig.savefig(
    OUT / "ggf_matched75_interpretability_compact.pdf",
    metadata={"CreationDate": None, "ModDate": None},
)
fig.savefig(OUT / "ggf_matched75_interpretability_compact.png", dpi=240)
(OUT / "ggf_matched75_interpretability_compact_labels.json").write_text(
    json.dumps(labels, indent=2), encoding="utf8"
)
plt.close(fig)
