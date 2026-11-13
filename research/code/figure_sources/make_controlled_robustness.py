from pathlib import Path
import numpy as np, pandas as pd, json, hashlib, shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = Path(__file__).resolve().parent.parent
INPUT = BASE / "figure_sources/inputs/controlled_robustness"
OUT = INPUT
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)
for filename, digest in json.loads((INPUT / "INPUT_SHA256.json").read_text()).items():
    assert hashlib.sha256((INPUT / filename).read_bytes()).hexdigest() == digest, (
        filename
    )
names = {
    "agf_g": "GGF",
    "supervised_vanilla": "Supervised Vanilla",
    "released_vanilla": "Vanilla BSF",
    "slics": "SLiCS",
    "concept_sae": "Concept-SAE",
}
colors = {
    "agf_g": "#008c82",
    "supervised_vanilla": "#c28d29",
    "released_vanilla": "#78878d",
    "slics": "#4778b4",
    "concept_sae": "#8a58ac",
}
methods = ["released_vanilla", "slics", "concept_sae", "supervised_vanilla", "agf_g"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 6.6,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    }
)
c = pd.read_csv(INPUT / "condition_metrics.csv").dropna(subset=["category_AP"])
per = (
    c.groupby(["model", "seed", "attribute", "level"]).category_AP.mean().reset_index()
)
cap = pd.read_csv(INPUT / "capture_by_category.csv")
cap = cap[
    (cap.group_budget == 1) & cap.attribute.isin(["rotation", "scale", "brightness"])
]
cp = cap.groupby(["model", "seed", "attribute"]).capture.mean().reset_index()
fig, axes = plt.subplots(
    1, 4, figsize=(6.875, 1.86), gridspec_kw={"width_ratios": [1, 1, 1, 1.12]}
)
for ax, attr, label in zip(
    axes[:3],
    ["rotation", "scale", "brightness"],
    ["Rotation (degrees)", "Object scale", "Object brightness"],
):
    for m in methods:
        a = (
            per[(per.model == m) & (per.attribute == attr)]
            .groupby("level")
            .category_AP.agg(["mean", "std"])
        )
        x = a.index.to_numpy()
        y = a["mean"].to_numpy() * 100
        sd = a["std"].to_numpy() * 100
        ax.plot(
            x,
            y,
            color=colors[m],
            marker="o" if m == "agf_g" else None,
            ms=2,
            lw=1.25 if m == "agf_g" else 0.9,
            ls="--" if m == "supervised_vanilla" else "-",
            label=names[m],
        )
        ax.fill_between(x, y - sd, y + sd, color=colors[m], alpha=0.12, lw=0)
    ax.set_xlabel(label, labelpad=3)
    ax.set_ylim(40, 101)
    ax.set_yticks([40, 60, 80, 100])
    ax.grid(axis="y", alpha=0.13, lw=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    if attr == "rotation":
        ax.set_ylabel("Object-category AP (%)")
    else:
        ax.set_yticklabels([])
axes[0].set_title("(a) Recognition across transformations", loc="left", pad=7)
ax = axes[3]
for j, m in enumerate(["released_vanilla", "supervised_vanilla", "agf_g"]):
    a = (
        cp[cp.model == m]
        .groupby("attribute")
        .capture.agg(["mean", "std"])
        .reindex(["rotation", "scale", "brightness"])
    )
    x = np.arange(3) + (j - 1) * 0.23
    ax.bar(
        x,
        a["mean"] * 100,
        0.21,
        color=colors[m],
        yerr=a["std"] * 100,
        capsize=1.5,
        error_kw={"elinewidth": 0.55, "capthick": 0.55},
    )
ax.set_xticks(range(3), ["Rot.", "Scale", "Bright."])
ax.set_ylim(0, 21)
ax.set_yticks([0, 5, 10, 15, 20])
ax.set_ylabel("Single-group Capture (%)", labelpad=2)
ax.set_title("(b) Regional information", loc="left", pad=7)
ax.grid(axis="y", alpha=0.13, lw=0.5)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    ncol=5,
    frameon=False,
    fontsize=6.5,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.015),
    handlelength=1.5,
    columnspacing=1.1,
)
fig.subplots_adjust(left=0.065, right=0.99, bottom=0.28, top=0.83, wspace=0.26)
for ext in ["pdf", "png", "svg"]:
    fig.savefig(FIG / ("ggf_controlled_robustness." + ext), dpi=500)
plt.close(fig)

# Repackage only measured tiles. The original figure pixels are retained.
models = ["agf_g", "supervised_vanilla", "released_vanilla"]


def tile(model, cat, row, col):
    im = Image.open(INPUT / f"{model}_21_{cat}.png").convert("RGB")
    return np.asarray(
        im.crop((39 + 315 * col, 109 + 287 * row, 301 + 315 * col, 371 + 287 * row))
    )


for cat in ["person", "dog"]:
    for j in [0, 2, 4]:
        for m in models[1:]:
            assert np.array_equal(tile(models[0], cat, 0, j), tile(m, cat, 0, j)), (
                cat,
                m,
                j,
            )
fig = plt.figure(figsize=(6.875, 7.35), facecolor="white")
left = 0.19
right = 0.995
bottom = 0.13
top = 0.94
w = (right - left - 0.035) / 6
h = (top - bottom - 0.02) / 7
labels = [
    "Input",
    "GGF response",
    "GGF PCA",
    "Supervised\nVanilla response",
    "Supervised\nVanilla PCA",
    "Vanilla response",
    "Vanilla PCA",
]
for ri, label in enumerate(labels):
    fig.text(
        0.18,
        top - (ri + 0.5) * h,
        label,
        ha="right",
        va="center",
        fontsize=6.8,
        color="#163337" if ri in [1, 2] else "#333333",
    )
    for ci, cat in enumerate(["person", "dog"]):
        for jj, col in enumerate([0, 2, 4]):
            x = left + (ci * 3 + jj) * w + (0 if ci == 0 else 0.035)
            y = top - (ri + 1) * h
            ax = fig.add_axes([x, y, w - 0.007, h - 0.009])
            ax.axis("off")
            if ri == 0:
                m, row = "agf_g", 0
            else:
                m, row = models[(ri - 1) // 2], 1 + (ri - 1) % 2
            ax.imshow(tile(m, cat, row, col), interpolation="none")
            if ri == 0:
                ax.set_title(f"{[-30, 0, 30][jj]}°", fontsize=7, pad=2)
fig.text(
    left + 1.5 * w,
    0.982,
    "Person",
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
)
fig.text(
    left + 4.5 * w + 0.035,
    0.982,
    "Dog",
    ha="center",
    va="top",
    fontsize=9,
    fontweight="bold",
)
# Exact display rules from the saved controlled-analysis renderer.
from matplotlib.colors import LinearSegmentedColormap, Normalize

teal = np.array([0.0, 0.58, 0.53])
cmap = LinearSegmentedColormap.from_list(
    "controlled_response", [np.ones(3), 0.15 + 0.85 * teal]
)
fig.text(
    0.19,
    0.109,
    "Response: group norm / training reference",
    fontsize=6.6,
    color="#163337",
)
cax = fig.add_axes([0.70, 0.102, 0.25, 0.014])
fig.colorbar(
    plt.cm.ScalarMappable(norm=Normalize(0, 0.85), cmap=cmap),
    cax=cax,
    orientation="horizontal",
    ticks=[0, 0.425, 0.85],
)
cax.set_xticklabels(["0", "0.425", "0.85+"])
cax.tick_params(labelsize=6, length=2, pad=1)
fig.text(0.19, 0.055, "PCA: independent RGB coordinates", fontsize=6.6, color="#163337")
for j, label in enumerate(["PC1 (red)", "PC2 (green)", "PC3 (blue)"]):
    x = 0.55 + j * 0.149
    ax = fig.add_axes([x, 0.045, 0.125, 0.012])
    ramp = np.full((1, 256, 3), 0.5)
    ramp[0, :, j] = np.linspace(0, 1, 256)
    ax.imshow(ramp, aspect="auto", extent=[-2, 2, 0, 1], interpolation="none")
    ax.set_yticks([])
    ax.set_xticks([-2, 0, 2], ["-2", "0", "+2"])
    ax.tick_params(labelsize=5.8, length=1.5, pad=1)
    ax.set_title(label, fontsize=6, pad=2)
fig.text(
    0.19,
    0.016,
    "PCA ticks: training standard deviations. Coordinates beyond the endpoints are clipped.",
    fontsize=6,
    color="#333333",
)
for ext in ["pdf", "png"]:
    fig.savefig(FIG / ("ggf_controlled_views_pca." + ext), dpi=500)
plt.close(fig)

category = pd.read_csv(INPUT / "category_metrics.csv")
summ = category.groupby(["model", "category_name"]).worst_condition_AP.agg(
    ["mean", "std"]
)
model_order = [
    "released_vanilla",
    "supervised_vanilla",
    "slics",
    "concept_sae",
    "agf_g",
]


def score(a, b):
    return f"${100 * a:.2f}\\pm{100 * b:.2f}$"


rows = []
for cat in ["person", "car", "dog", "chair", "potted plant", "vase"]:
    rows.append(
        cat.capitalize()
        + " & "
        + " & ".join(score(*summ.loc[(m, cat)].values) for m in model_order)
        + r"\\"
    )
text = (
    r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{5pt}
\caption{Worst-condition object-category AP (\%, mean $\pm$ sample SD).
For each category, the minimum AP across the fifteen foreground
conditions is computed within a fitted model before aggregation.}
\label{tab:controlled-category-ap}
\begin{tabular}{@{}lrrrrr@{}}\toprule
Category & Vanilla & Supervised Vanilla & SLiCS & Concept-SAE & GGF\\\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule\end{tabular}\end{table*}
"""
)
(OUT / "table_controlled_categories.tex").write_text(text)
rows = []
for cat in ["person", "car", "dog", "chair", "potted plant", "vase"]:
    for attr in ["rotation", "scale", "brightness"]:
        ss = (
            cap[(cap.category == cat) & (cap.attribute == attr)]
            .groupby("model")
            .capture.agg(["mean", "std"])
        )
        rows.append(
            cat.capitalize()
            + " & "
            + attr.capitalize()
            + " & "
            + " & ".join(
                score(*ss.loc[m].values)
                for m in ["released_vanilla", "supervised_vanilla", "agf_g"]
            )
            + r"\\"
        )
text = (
    r"""\begin{table*}[t]\centering\small
\setlength{\tabcolsep}{6pt}
\caption{Controlled single-group Capture by category and transformation
(\%, mean $\pm$ sample SD). Squared errors and target energies pool over
objects and levels within each category before computing Capture.}
\label{tab:controlled-category-capture}
\begin{tabular}{@{}llrrr@{}}\toprule
Category & Transformation & Vanilla & Supervised Vanilla & GGF\\\midrule
"""
    + "\n".join(rows)
    + r"""
\bottomrule\end{tabular}\end{table*}
"""
)
(OUT / "table_controlled_capture.tex").write_text(text)
cp.to_csv(OUT / "controlled_capture_per_start.csv", index=False)
per.to_csv(OUT / "controlled_AP_per_start.csv", index=False)
print("Built two measured figures and complete category tables.")
print(cp.groupby(["model", "attribute"]).capture.agg(["mean", "std"]).to_string())
