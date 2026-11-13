"""New endpoint ranks; exact reuse of six completed 8D TRAIN rank analyses."""

import csv, gc, json, shutil, zipfile
from pathlib import Path
import numpy as np
import torch
from rank_core import atomic_json, block_statistics, fingerprint, sha, summarize
from rank_worker import analyze_model, across_seeds, csv_write
from coco_baselines import training
from coco_ggf import twelve_capacity as exp

RANK_RUN = "7c974de9e38700416635ab01ad903186d0375c01bdd9fcc91e519f61cd6eef0c"
RANK_SOURCE = "0405de40cabd71334d58bcd8ae6f411695de09e87548b988790b86d57ef40339"


def verify_source(ctx, path):
    folder = Path(path) / "results" / RANK_RUN[:20]
    protocol = json.loads((folder / "protocol.json").read_text())
    summary = json.loads((folder / "summary.json").read_text())
    cohort = json.loads((folder / "cohort.json").read_text())
    if (
        fingerprint(protocol) != RANK_RUN
        or summary["run_id"] != RANK_RUN
        or not summary["complete"]
        or len(summary["models"]) != 30
    ):
        raise ValueError("Finish the prior effective-rank notebook first")
    if (
        protocol["analysis_source_sha"] != RANK_SOURCE
        or protocol["capacity_run"] != exp.PARENT_RUN
        or protocol["normalization_sha"] != ctx["store"].config["normalization_sha"]
    ):
        raise ValueError("Completed rank source identity differs")
    if (
        protocol["train_ids"] != ctx["packs"]["train"]["ids"]
        or cohort["identity"] != ctx["packs"]["train"]["identity"]
    ):
        raise ValueError("Rank source TRAIN cohort differs")
    for row in ctx["roots"]:
        tag = row["arm"] + "_s" + str(row["seed"])
        sf = folder / "models" / tag
        done = json.loads((sf / "complete.json").read_text())
        expected = fingerprint(
            dict(
                protocol=RANK_RUN,
                train=cohort["identity"],
                model=row["weights_sha"],
                rank=8,
                k=16,
            )
        )
        if (
            done["signature"] != expected
            or not done["complete"]
            or done["patches"] != 392000
        ):
            raise ValueError("Saved 8D rank analysis identifies a different model")
        for name, h in done["file_hashes"].items():
            if Path(name).name != name or sha(sf / name) != h:
                raise ValueError("Saved 8D rank analysis file changed")
    return folder


def plot(folder, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for family in ["initialized_slots", "reserve_slots"]:
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), layout="constrained")
        for ax, key, title in zip(
            axes,
            [
                "code_effective_rank_mean",
                "contribution_effective_rank_mean",
                "code_utilization_mean",
            ],
            [
                "Coefficient effective rank",
                "Contribution effective rank",
                "Coefficient utilization",
            ],
        ):
            for arms, color, name, offset in [
                (["agf_8d16", "agf_8d16next", "agf_12d16"], "#007f83", "AGF", -0.12),
                (
                    ["vanilla_8d16", "vanilla_8d16next", "vanilla_12d16"],
                    "#6b5794",
                    "Vanilla BSF",
                    0.12,
                ),
            ]:
                for i, arm in enumerate(arms):
                    rr = [
                        r
                        for r in rows
                        if r["arm"] == arm
                        and r["slot_family"] == family
                        and r["sample_subset"] == "at_least_32_active_patches"
                    ]
                    values = [r[key] for r in rr if r[key] is not None]
                    if values:
                        ax.scatter(
                            np.full(len(values), i + offset),
                            values,
                            color=color,
                            s=22,
                            alpha=0.65,
                        )
                        ax.scatter(
                            i + offset,
                            np.mean(values),
                            marker="_",
                            s=65,
                            linewidths=2.4,
                            color=color,
                            label=name if i == 0 else None,
                        )
            ax.set_xticks(range(3), ["Saved 8D", "Continued 8D", "12D"], rotation=15)
            ax.set_title(title, fontsize=10)
            ax.set_ylim(bottom=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.15)
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle(
            (
                "First 75 slots; semantic initialization only in AGF"
                if family == "initialized_slots"
                else "949 reserve slots"
            )
            + "\nTRAIN-only; dots are seeds",
            fontsize=10,
        )
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ["png", "pdf"]:
            fig.savefig(folder / (family + "." + ext), dpi=220)
        plt.close(fig)


def run(ctx, path):
    source = verify_source(ctx, path)
    frozen = exp.verify_freeze(ctx)
    folder = ctx["store"].results / "effective_rank"
    folder.mkdir(exist_ok=True)
    x = ctx["packs"]["train"]["x"].reshape(-1, 768)
    signature = fingerprint(
        dict(
            experiment=ctx["store"].run_id,
            train=ctx["packs"]["train"]["identity"],
            rank_source=RANK_RUN,
            algorithm="centered conditional coefficient and contribution covariance",
            sample_threshold=32,
        )
    )
    rows = []
    blocks = []
    count = 0
    for row in ctx["roots"] + frozen["models"]:
        tag = row["arm"] + "_s" + str(row["seed"])
        dest = folder / "models" / tag
        dest.mkdir(parents=True, exist_ok=True)
        if row["arm"] in exp.SOURCE_ARMS:
            original = source / "models" / tag
            for file in original.iterdir():
                if file.is_file():
                    shutil.copyfile(file, dest / file.name)
            rr = json.loads((dest / "summary.json").read_text())
            bb = json.loads((dest / "blocks.json").read_text())
            print("Reused completed 8D rank analysis:", tag, flush=True)
        else:
            model_sig = fingerprint(
                dict(
                    run=signature,
                    weights=row["weights_sha"],
                    rank=row["rank"],
                    k=row["active_groups"],
                )
            )
            completion = dest / "complete.json"
            if completion.exists():
                done = json.loads(completion.read_text())
                if done["signature"] != model_sig or not done["complete"]:
                    raise ValueError("New rank analysis fingerprint changed")
                for file, h in done["file_hashes"].items():
                    if Path(file).name != file or sha(dest / file) != h:
                        raise ValueError("New rank result checksum differs")
                rr = json.loads((dest / "summary.json").read_text())
                bb = json.loads((dest / "blocks.json").read_text())
            else:
                state = exp.memory.required(ctx["store"], row["key"])
                exp.validate_state(ctx, row, state)
                cfg = exp.subcontext(ctx, row["rank"], 16)["base"]
                model = training.make_model("vanilla", cfg).to(ctx["device"])
                model.load_state_dict(state["weights"])
                decoder = (
                    state["weights"]["W_dec"]
                    .reshape(ctx["base"]["groups"], row["rank"], 768)
                    .double()
                    .numpy()
                    .copy()
                )
                del state
                gc.collect()
                print("New endpoint TRAIN rank inference:", tag, flush=True)
                moments = analyze_model(
                    model, x, folder / "accumulators" / tag, model_sig, ctx["device"]
                )
                bb, arrays = block_statistics(moments, decoder)
                meta = dict(
                    arm=row["arm"],
                    seed=row["seed"],
                    rank=row["rank"],
                    active_groups=16,
                    weights_sha=row["weights_sha"],
                    scheduled_updates=exp.TOTAL,
                    semantic_initialization=row["supervised"],
                )
                for r in bb:
                    r.update(meta)
                    r["initialized_category"] = (
                        row["targets"][r["group"]]["category"]
                        if row["supervised"] and r["group"] < 75
                        else None
                    )
                rr = [dict(r, **meta) for r in summarize(bb)]
                atomic_json(dest / "blocks.json", bb)
                csv_write(dest / "blocks.csv", bb)
                atomic_json(dest / "summary.json", rr)
                with (dest / "spectra.npz").open("wb") as f:
                    np.savez_compressed(f, **arrays)
                hashes = {
                    p.name: sha(p) for p in dest.iterdir() if p.name != "complete.json"
                }
                atomic_json(
                    completion,
                    dict(
                        signature=model_sig,
                        complete=True,
                        file_hashes=hashes,
                        patches=len(x),
                    ),
                )
                del model, decoder
                gc.collect()
                torch.cuda.empty_cache()
        rows.extend(rr)
        blocks.extend(bb)
        count += 1
        atomic_json(
            folder / "progress.json",
            dict(complete=False, completed_models=count, total_models=18),
        )
        csv_write(folder / "model_summaries.csv", rows)
    means = across_seeds(rows)
    csv_write(folder / "three_seed_means.csv", means)
    csv_write(folder / "block_statistics.csv", blocks)
    plot(folder / "figures", rows)
    result = dict(
        complete=True,
        new_models_analyzed=12,
        saved_8d_models_reused=6,
        source_rank_run=RANK_RUN,
        train_images=2000,
        patches=392000,
        new_gradient_updates=0,
        new_test_inference=False,
        three_seed_means=means,
        scope="Empirical active-code variation; not ground-truth concept dimensionality or MDL efficiency.",
    )
    atomic_json(folder / "summary.json", result)
    atomic_json(
        folder / "progress.json",
        dict(complete=True, completed_models=18, total_models=18),
    )
    exp.matched.assert_sources(ctx)
    print(
        "All 12 new rank analyses complete; six existing 8D analyses reused.",
        flush=True,
    )
    return result
