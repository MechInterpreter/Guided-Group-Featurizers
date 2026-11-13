"""Rebuild measured comparisons and resources from immutable run records."""

from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

H = Path(__file__).resolve().parent.parent
(H / "sec").mkdir(exist_ok=True)
(H / "figures").mkdir(exist_ok=True)
D = H / "evidence/voc_slics_final"
OLD = H / "figure_sources/inputs/final_comparisons"
OUT = H / "figures"
V = pd.read_csv(D / "VOC_completed_per_seed.csv")
P = pd.read_csv(D / "compute_profiles.csv").set_index("name")
A = json.loads((D / "verified_analysis.json").read_text())
legacy = json.loads((D / "legacy_resources.json").read_text())
assert len(V) == 6 and set(V.seed) == {21, 22, 23} and set(V.method) == {"GGF", "SLiCS"}
assert len(P) == 17 and not P.full_FLOPs_available.any()
for name, digest in json.loads((D / "SHA256.json").read_text()).items():
    assert hashlib.sha256((D / name).read_bytes()).hexdigest() == digest, name
for method in ["GGF", "SLiCS"]:
    for metric in ["Capture", "patch_AP", "VE", "training_loop_minutes"]:
        assert np.isclose(
            V[V.method == method][metric].mean(), A["summary"][method][metric]["mean"]
        )
C = {
    "Vanilla BSF": "#8b999e",
    "Supervised Vanilla": "#c08a25",
    "GGF": "#00877b",
    "Concept-SAE": "#7b5aa6",
    "SLiCS": "#567ba4",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.labelsize": 7.3,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.5,
        "savefig.facecolor": "white",
        "svg.hashsalt": "GGF-VOC-SLiCS-final-v1",
    }
)


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        metadata = (
            {"CreationDate": None, "ModDate": None}
            if ext == "pdf"
            else {"Date": None}
            if ext == "svg"
            else {}
        )
        fig.savefig(OUT / (name + "." + ext), dpi=360, metadata=metadata)
    plt.close(fig)


def meanrows(f, methods):
    return [
        dict(
            method=m,
            **{
                k: dict(
                    mean=float(f[f.method == m][k].mean()),
                    sd=float(f[f.method == m][k].std(ddof=1)),
                )
                for k in ["capture", "patch_AP", "VE"]
            },
        )
        for m in methods
    ]


blocks = []
common = {}
for dataset in ["coco", "voc"]:
    f = pd.read_csv(OLD / (dataset + "_per_run.csv"))
    f["method"] = f.method.str.replace(r"^Concept-SAE.*", "Concept-SAE", regex=True)
    assert len(f) == 12 and (f.groupby("method").size() == 3).all()
    common[dataset] = f
    blocks.append(
        (
            dataset.upper() + ": 1,024-group BSFs and Concept-SAE",
            meanrows(f, ["Vanilla BSF", "Supervised Vanilla", "GGF", "Concept-SAE"]),
        )
    )
f = pd.read_csv(OLD / "slics_means.csv")
oldrows = []
for arm, m in [("slics_final_partial", "SLiCS"), ("matched_ggf_final", "GGF")]:
    q = f[(f.arm == arm) & (f.seed_group == "all_five")].iloc[0]
    oldrows.append(
        dict(
            method=m,
            **{
                k: dict(mean=float(q[col]), sd=float(q[col + "_sd"]))
                for k, col in [
                    ("capture", "capture_pct"),
                    ("patch_AP", "patch_AP_pct"),
                    ("VE", "VE_pct"),
                ]
            },
        )
    )
blocks.append(("COCO: 75-group SLiCS comparison", oldrows))
blocks.append(
    (
        "VOC: 19-group SLiCS comparison",
        meanrows(V.rename(columns={"Capture": "capture"}), ["SLiCS", "GGF"]),
    )
)
rows = []
headers = []
y = 0
for label, rr in blocks:
    headers.append((y, label))
    y += 0.82
    for r in rr:
        rows.append(dict(r, y=y))
        y += 0.84
    y += 0.18
fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.25), sharey=True)
fig.subplots_adjust(left=0.205, right=0.986, bottom=0.14, top=0.965, wspace=0.115)
for i, (ax, key, title, limit, ticks) in enumerate(
    zip(
        axs,
        ["capture", "patch_AP", "VE"],
        ["Capture (%)", "Patch AP (%)", "Reconstruction VE (%)"],
        [60, 117, 107],
        [[0, 20, 40], [0, 50, 100], [0, 50, 100]],
    )
):
    for r in rows:
        m = r[key]["mean"]
        sd = r[key]["sd"]
        ax.barh(
            r["y"],
            m,
            height=0.55,
            color=C[r["method"]],
            xerr=sd,
            error_kw=dict(ecolor="#243940", elinewidth=0.5, capsize=1, capthick=0.5),
            zorder=3,
        )
        ax.text(
            m + sd + limit * 0.011,
            r["y"],
            f"{m:.2f}",
            va="center",
            fontsize=6.8,
            color="#172a2e",
        )
    ax.set_ylim(y - 0.25, -0.35)
    ax.set_xlim(0, limit)
    ax.set_xlabel(title, labelpad=1.5)
    ax.set_yticks([r["y"] for r in rows])
    ax.set_yticklabels([r["method"] for r in rows], fontsize=6.9)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", length=2, pad=1.5)
    ax.grid(axis="x", color="#e6ebed", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    for yy, _ in headers[1:]:
        ax.axhline(yy - 0.30, color="#c4ced1", linewidth=0.45)
    if i:
        ax.tick_params(axis="y", labelleft=False)
for yy, label in headers:
    axs[0].text(
        -0.255,
        yy,
        label,
        transform=axs[0].get_yaxis_transform(),
        fontsize=6.9,
        fontweight="bold",
        color="#243f48",
        ha="left",
        va="center",
        clip_on=False,
    )
save(fig, "agf_released_bars")
# Each resource column keeps its own device and protocol.
q = (
    common["voc"]
    .groupby("method")
    .agg({"patch_AP": ["mean", "std"], "parameters": ["mean"]})
)
cohorts = [
    dict(
        title="VOC: 1,024-group BSFs / T4",
        rows=[
            dict(
                method=m,
                params=float(q.loc[m, ("parameters", "mean")]) / 1e6,
                ap=float(q.loc[m, ("patch_AP", "mean")]),
                ap_sd=float(q.loc[m, ("patch_AP", "std")]),
                time=legacy["voc_T4"][m]["minutes"],
                time_sd=legacy["voc_T4"][m]["sd"],
            )
            for m in ["Vanilla BSF", "Supervised Vanilla", "GGF", "Concept-SAE"]
        ],
    )
]
cohorts.append(
    dict(
        title="COCO: 75 groups / T4",
        rows=[
            dict(
                method=r["method"],
                params=legacy["coco75_T4"][r["method"]]["parameters"] / 1e6,
                ap=r["patch_AP"]["mean"],
                ap_sd=r["patch_AP"]["sd"],
                time=legacy["coco75_T4"][r["method"]]["minutes"],
                time_sd=legacy["coco75_T4"][r["method"]]["sd"],
            )
            for r in oldrows
        ],
    )
)
cohorts.append(
    dict(
        title="VOC: 19 groups / A100",
        rows=[
            dict(
                method=m,
                params=float(V[V.method == m].learned_parameters.iloc[0]) / 1e6,
                ap=float(V[V.method == m].patch_AP.mean()),
                ap_sd=float(V[V.method == m].patch_AP.std(ddof=1)),
                time=float(V[V.method == m].training_loop_minutes.mean()),
                time_sd=float(V[V.method == m].training_loop_minutes.std(ddof=1)),
            )
            for m in ["GGF", "SLiCS"]
        ],
    )
)
fig, axs = plt.subplots(2, 3, figsize=(6.875, 3.60))
fig.subplots_adjust(
    left=0.075, right=0.985, bottom=0.12, top=0.92, hspace=0.42, wspace=0.38
)
for j, c in enumerate(cohorts):
    for i, metric in enumerate(["params", "time"]):
        ax = axs[i, j]
        ax.set_xscale("log")
        for r in c["rows"]:
            ax.errorbar(
                r[metric],
                r["ap"],
                yerr=r["ap_sd"],
                xerr=r["time_sd"] if metric == "time" else None,
                fmt="o",
                color=C[r["method"]],
                ms=4,
                capsize=2,
                elinewidth=0.6,
            )
            label = {
                "Vanilla BSF": "Vanilla",
                "Supervised Vanilla": "Sup. Vanilla",
            }.get(r["method"], r["method"])
            offset = (4, 3)
            if j == 0 and r["method"] == "Supervised Vanilla":
                offset = (5, -8)
            if j == 0 and r["method"] == "GGF":
                offset = (4, 7)
            if r["method"] == "Concept-SAE":
                offset = (-45, 6)
            ax.annotate(
                label,
                (r[metric], r["ap"]),
                xytext=offset,
                textcoords="offset points",
                fontsize=6.8,
            )
        ax.grid(alpha=0.18, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel(
            "Learned parameters (M)"
            if metric == "params"
            else "Recorded update loops (min)"
        )
        ax.set_ylim((65, 103) if j == 0 else (43, 92) if j == 1 else (60, 96))
        if j == 0:
            ax.set_ylabel("Patch AP (%)")
        if i == 0:
            ax.set_title(c["title"], fontsize=7.8, pad=5)
        vals = [r[metric] for r in c["rows"]]
        ax.set_xlim(min(vals) * 0.65, max(vals) * 1.75)
        ticks = (
            [[20, 50, 100], [0.5, 1, 2], [0.15, 0.3, 0.6]]
            if i == 0
            else [[5, 10, 20], [10, 50, 100], [2, 10, 50]]
        )[j]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
save(fig, "ggf_final_resources")
specs = [
    ("released_vanilla1024_inference", "Vanilla (1,024)", "Vanilla BSF"),
    (
        "supervised_vanilla1024_inference",
        "Supervised Vanilla (1,024)",
        "Supervised Vanilla",
    ),
    ("agf_g1024_inference", "GGF (1,024)", "GGF"),
    ("ConceptSAE_inference", "Concept-SAE", "Concept-SAE"),
    ("GGF19_inference", "GGF (19)", "GGF"),
    ("SLiCS19_partial", "SLiCS (19), category NNLS", "SLiCS"),
    ("SLiCS19_joint", "SLiCS (19), joint NNLS", "SLiCS"),
]
fig, axs = plt.subplots(1, 3, figsize=(6.875, 2.60), sharey=True)
fig.subplots_adjust(left=0.275, right=0.985, bottom=0.22, top=0.90, wspace=0.32)
for i, (name, label, method) in enumerate(specs):
    r = P.loc[name]
    yy = i + (0.65 if i >= 4 else 0)
    color = C[method]
    axs[0].errorbar(
        r.seconds_mean * 1000,
        yy,
        xerr=r.seconds_SD * 1000,
        fmt="o",
        color=color,
        ms=4,
        capsize=2,
        elinewidth=0.65,
    )
    axs[1].plot([r.allocated_GiB, r.reserved_GiB], [yy, yy], color="#c2cbce", lw=1)
    axs[1].plot(r.allocated_GiB, yy, "o", color=color, ms=4)
    axs[1].plot(r.reserved_GiB, yy, "o", mfc="white", mec=color, ms=4)
    axs[2].plot(r.FLOPs_per_image / 1e9, yy, "o", color=color, ms=4)
    for ax, value in [
        (axs[0], r.seconds_mean * 1000),
        (axs[2], r.FLOPs_per_image / 1e9),
    ]:
        ax.annotate(
            f"{value:.2f}",
            (value, yy),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.6,
        )
for ax in axs:
    ax.set_ylim(7.2, -0.5)
    ax.grid(axis="x", alpha=0.18, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    ax.axhline(3.83, color="#b7c7ca", lw=0.55)
    ax.spines["left"].set_visible(False)
axs[0].set_xscale("log")
axs[0].set_xlim(2.8, 280)
axs[0].set_xlabel("Call time (ms / 8 images)")
axs[0].set_title("A100 inference latency")
axs[1].set_xlim(0, 0.56)
axs[1].set_xlabel("Peak GPU memory (GiB)")
axs[1].set_title("GPU memory")
axs[2].set_xscale("log")
axs[2].set_xlim(0.09, 60)
axs[2].set_xlabel("Registered GFLOPs / image")
axs[2].set_title("Registered operations")
axs[0].set_yticks([i + (0.65 if i >= 4 else 0) for i in range(7)])
axs[0].set_yticklabels([s[1] for s in specs], fontsize=7)
for ax in axs[1:]:
    ax.tick_params(axis="y", labelleft=False)
for ax, ticks in [(axs[0], [5, 20, 100]), (axs[2], [0.1, 1, 10])]:
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
save(fig, "ggf_execution_profiles")


def pm(m, s, n=2):
    return "$" + f"{m:.{n}f}\\pm{s:.{n}f}" + "$"


def tab(name, caption, label, columns, header, body, wide=False):
    lines = [
        r"\begin{table*}[t]\centering\small"
        if wide
        else r"\begin{table}[htbp]\centering\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{@{}" + columns + r"@{}}\toprule",
        header + r"\\\midrule",
        *body,
        r"\bottomrule\end{tabular}",
        r"\end{table*}" if wide else r"\end{table}",
    ]
    (H / "sec" / name).write_text("\n".join(lines) + "\n")


body = []
for seed in [21, 22, 23]:
    for m in ["GGF", "SLiCS"]:
        r = V[(V.seed == seed) & (V.method == m)].iloc[0]
        body.append(
            f"{seed} & {m} & {r.Capture:.2f} & {r.patch_AP:.2f} & {r.VE:.2f}" + r"\\"
        )
body.append(r"\midrule")
for m in ["GGF", "SLiCS"]:
    r = A["summary"][m]
    body.append(
        "Mean & "
        + m
        + " & "
        + " & ".join(
            pm(r[k]["mean"], r[k]["sd"]) for k in ["Capture", "patch_AP", "VE"]
        )
        + r"\\"
    )
tab(
    "table_voc_slics.tex",
    r"VOC category-matched comparison (\%). Both methods use 19 twelve-direction groups and 10,640 updates. Aggregates show mean $\pm$ sample SD.",
    "tab:voc-slics-final",
    "llrrr",
    "Seed & Method & Capture & Patch AP & VE",
    body,
)
body = [
    dict(Capture="Capture", patch_AP="Patch AP", VE="VE")[r["metric"]]
    + " & "
    + pm(r["GGF_minus_SLiCS"], r["sample_SD"])
    + f" & [{r['CI95_low']:.2f}, {r['CI95_high']:.2f}]"
    + r"\\"
    for r in A["paired"]
]
tab(
    "table_voc_slics_gains.tex",
    r"Paired VOC GGF-minus-SLiCS gains in percentage points. SD and Student-$t$ intervals describe paired training-seed differences.",
    "tab:voc-slics-gains",
    "lrr",
    r"Metric & Gain $\pm$ SD & 95\% interval",
    body,
)
labels = {
    "GGF19_inference": "GGF (19), inference",
    "GGF19_training_plain": "GGF (19), ordinary update",
    "GGF19_training_controller": "GGF (19), controller update",
    "SLiCS19_partial": "SLiCS (19), category NNLS",
    "SLiCS19_joint": "SLiCS (19), joint NNLS",
    "SLiCS19_training": "SLiCS (19), training update",
    "agf_g1024_inference": "GGF (1,024), inference",
    "agf_g1024_training_plain": "GGF (1,024), ordinary update",
    "agf_g1024_training_controller": "GGF (1,024), controller update",
    "released_vanilla1024_inference": "Vanilla (1,024), inference",
    "released_vanilla1024_training_plain": "Vanilla (1,024), training update",
    "supervised_vanilla1024_inference": "Supervised Vanilla (1,024), inference",
    "supervised_vanilla1024_training_plain": "Supervised Vanilla (1,024), training update",
    "ConceptSAE_inference": "Concept-SAE, inference",
    "ConceptSAE_training_stage1": "Concept-SAE, stage 1 update",
    "ConceptSAE_training_stage2": "Concept-SAE, stage 2 update",
    "ConceptSAE_training_stage3": "Concept-SAE, stage 3 update",
}
body = [
    f"{labels[n]} & {int(r.images)} & "
    + pm(r.seconds_mean * 1000, r.seconds_SD * 1000)
    + f" & {r.allocated_GiB:.3f}/{r.reserved_GiB:.3f} & {r.FLOPs_per_image / 1e9:.3f}"
    + r"\\"
    for n, r in P.iterrows()
]
tab(
    "table_execution_profiles.tex",
    r"A100 execution profiles on VOC training batches, excluding the frozen DINOv3 backbone. Times are mean $\pm$ sample SD over three independent calls. GPU peaks are maximum allocated/reserved GiB. Registered GFLOPs use one instrumented call with fused multiply-add counted as two operations. These are partial counts, not full FLOPs.",
    "tab:execution-profiles",
    "lrrrr",
    "Method and phase & Images & Call time (ms) & GPU peaks (GiB) & Registered GFLOPs/image",
    body,
    True,
)
body = [r"\multicolumn{4}{l}{T4: 1,024 groups for BSF arms}\\"]
for m in ["Vanilla BSF", "Supervised Vanilla", "GGF", "Concept-SAE"]:
    r = legacy["voc_T4"][m]
    body.append(
        f"{m} & {r['parameters'] / 1e6:.2f} & "
        + pm(r["minutes"], r["sd"])
        + f" & {r['allocated']:.2f}/{r['reserved']:.2f}"
        + r"\\"
    )
body.append(r"\midrule\multicolumn{4}{l}{A100: 19-group SLiCS comparison}\\")
for m in ["GGF", "SLiCS"]:
    q = A["summary"][m]["training_loop_minutes"]
    r = A["fit_memory_max"][m]
    n = int(V[V.method == m].learned_parameters.iloc[0])
    body.append(
        f"{m} & {n / 1e6:.3f} & "
        + pm(q["mean"], q["sd"])
        + f" & {r['fit_peak_allocated_GiB']:.3f}/{r['fit_peak_reserved_GiB']:.3f}"
        + r"\\"
    )
tab(
    "table_compute.tex",
    r"VOC fitting resources under two protocols. Update loops: minutes, mean $\pm$ sample SD. GPU peaks: maximum allocated/reserved GiB across fits. Devices and group counts differ between blocks (Appendix~\ref{app:compute-reporting}).",
    "tab:compute-main",
    "lrrr",
    "Method & Params (M) & Update min & GPU peaks",
    body,
)
body = [
    f"{m} & {legacy['coco75_T4'][m]['parameters']:,} & "
    + pm(legacy["coco75_T4"][m]["minutes"], legacy["coco75_T4"][m]["sd"])
    + r"\\"
    for m in ["GGF", "SLiCS"]
]
tab(
    "table_coco_slics_cost.tex",
    r"COCO 75-group T4 fitting resources. Loop time is mean $\pm$ sample SD in minutes. GPU peaks were not recorded.",
    "tab:coco-slics-cost",
    "lrr",
    "Method & Parameters & Update min",
    body,
)
body = []
for m in ["GGF", "SLiCS"]:
    r = A["summary"][m]
    body.append(
        m
        + " & "
        + pm(r["training_loop_minutes"]["mean"], r["training_loop_minutes"]["sd"])
        + " & "
        + pm(
            r["fit_invocation_seconds"]["mean"] / 60,
            r["fit_invocation_seconds"]["sd"] / 60,
        )
        + r"\\"
    )
tab(
    "table_voc_fit_wall.tex",
    r"VOC A100 fit-invocation time in minutes (mean $\pm$ sample SD). This includes initialization and development checks but excludes final evaluation.",
    "tab:voc-fit-wall",
    "lrr",
    "Method & Update loops & Fit invocation",
    body,
)
audit = dict(
    figure3_blocks=blocks,
    resource_cohorts=cohorts,
    full_FLOPs_available=False,
    sources={
        p.relative_to(H).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in [D, OLD]
        for p in folder.glob("*")
        if p.is_file()
    },
)
(OUT / "final_voc_integration_audit.json").write_text(json.dumps(audit, indent=2))
(OUT / "figure3_audit.json").write_text(
    json.dumps(
        dict(
            rows=rows,
            headers=headers,
            sources=audit["sources"],
            uncertainty="Sample SD within each protocol, without pooling cohorts",
        ),
        indent=2,
    )
)
print(
    "Generated supervised-comparison, resource and inference-profile plots, and six tables."
)
