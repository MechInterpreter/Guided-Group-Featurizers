"""Fixed annotation-chosen examples, calibration-chosen groups, no outcome cherry-picking."""

import csv
import html
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from coco_baselines import data
from coco_baselines.storage import sha_file
from coco_adaptation import core, coverage


def select_examples(ids, evidence, category_indices, seed, count=1):
    usable = evidence.all(2)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in category_indices:
        candidates = np.flatnonzero(usable[:, c])
        take = rng.permutation(candidates)[:count]
        chosen.extend(dict(category_index=int(c), image_id=int(ids[i])) for i in take)
    return chosen


def freeze_gallery(master, coco, plan, cfg, scan_store=None):
    old = master.load("gallery_examples")
    if old is not None:
        return old
    # Annotation-only choice before any model training, using exactly the planned crop.
    from .dataset import candidate_ids

    all_ids = candidate_ids(
        coco, cfg["split_seed"], cfg["candidate_images_per_category"]
    )
    index = {i: j for j, i in enumerate(all_ids)}
    rows = {}
    wanted = set(plan["splits"]["test"])
    for lo in range(0, len(all_ids), cfg["scan_chunk"]):
        part = (scan_store or master).load(f"coverage_scan_{lo:06d}")
        if part is None:
            raise ValueError("Missing coverage scan")
        for i, e in zip(part["ids"], part["evidence"]):
            if i in wanted:
                rows[i] = e
    ids = plan["splits"]["test"]
    evidence = np.stack([rows[i] for i in ids])
    selected = select_examples(
        ids,
        evidence,
        cfg["category_indices"],
        cfg["gallery_seed"],
        cfg["gallery_per_category"],
    )
    master.save("gallery_examples", selected)
    master.write(
        "gallery_selection.json",
        dict(
            examples=selected,
            rule="One random eligible image per supported category, fixed before training. No activation, score or visual quality is used to choose images.",
        ),
    )
    return selected


@torch.no_grad()
def group_heat(model, x, device):
    model.eval()
    z = model.encode(torch.as_tensor(x, device=device)).reshape(
        len(x), model.n_groups, model.group_size
    )
    return z.norm(dim=-1).float().cpu().numpy()


def render(master, branches, packs, test, coco, cfg, local, provenance):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    examples = master.load("gallery_examples")
    index = {i: j for j, i in enumerate(test["ids"])}
    wanted = sorted({e["image_id"] for e in examples})
    positions = [index[i] for i in wanted]
    position = {i: j for j, i in enumerate(wanted)}
    folder = master.results / "heatmaps"
    folder.mkdir(exist_ok=True)
    rows = []
    figures = []
    names = {
        "native": "Native BSF",
        "supervised_fixed_equal_search_compute": "AGF without reorganization",
        "supervised_adaptive": "AGF with reorganization enabled",
    }
    for branch, b in branches.items():
        saved = master.load("gallery_done_" + branch)
        if saved is not None and all(
            (folder / name).exists() and sha_file(folder / name) == digest
            for name, digest in saved["files"].items()
        ):
            rows.extend(saved["rows"])
            figures.extend(saved["figures"])
            master.event("heatmap_branch_reused", branch=branch)
            continue
        row_start = len(rows)
        figure_start = len(figures)
        store = b["store"]
        models = {}
        scales = {}
        maps = {}
        arms = {a["arm"]: a for a in b["arms"]}
        for arm in names:
            a = arms[arm]
            state = store.load(a["key"])
            model = core.new_model(
                a["method"], b["native"].config["config"], state["best"], cfg["device"]
            )
            hh = []
            for i in positions:
                hh.append(group_heat(model, test["x"][i], cfg["device"]))
            models[arm] = np.stack(hh)
            mapping = store.load("mapping_" + a["key"])
            maps[arm] = {r["category_index"]: r for r in mapping["mappings"]}
            # Calibration-only saturation scale: no per-image auto-contrast that hides inactive groups.
            scale_key = "gallery_scales_" + a["key"]
            q = store.load(scale_key)
            if q is None:
                group_ids = sorted(
                    {
                        r["detector"]["group"]
                        for r in mapping["mappings"]
                        if r["detector"]
                    }
                )
                hh = []
                for image_x in packs["mapping"]["x"]:
                    hh.append(group_heat(model, image_x, cfg["device"])[:, group_ids])
                heat = np.concatenate(hh)
                q = {
                    g: max(float(np.quantile(heat[:, j], 0.99)), 1e-8)
                    for j, g in enumerate(group_ids)
                }
                store.save(scale_key, q)
            scales[arm] = q
            del model
        np.savez_compressed(
            folder / (branch + "_all_group_heat.npz"),
            image_ids=np.asarray(wanted),
            **models,
        )
        for e in examples:
            c = e["category_index"]
            image_id = e["image_id"]
            i = index[image_id]
            category = provenance["categories"][c]["name"]
            im = coco.imgs[image_id]
            path = data.download(
                data.COCO_BUCKET + "train2017/" + im["file_name"],
                Path(local) / "images" / im["file_name"],
            )
            with Image.open(path) as image:
                crop = np.asarray(
                    data.transform_image(
                        image.convert("RGB"), data.geometry(*image.size)
                    )
                )
            mask = (
                test["masks"][i]["fraction"][:, c].reshape(14, 14).astype(float) / 255
            )
            shared_scale = max(
                scales[arm][maps[arm][c]["detector"]["group"]] for arm in names
            )
            fig, axes = plt.subplots(2, 4, figsize=(15, 7))
            axes[0, 0].imshow(crop)
            axes[0, 0].imshow(
                mask,
                extent=(0, 224, 224, 0),
                vmin=0,
                vmax=1,
                cmap="Greens",
                alpha=0.45,
                interpolation="nearest",
            )
            axes[0, 0].set_title(f"{category}: annotation mask\nimage {image_id}")
            axes[1, 0].imshow(crop)
            axes[1, 0].set_title("Exact DINO crop\n14 × 14 patch grid")
            for line in range(16, 224, 16):
                axes[1, 0].axhline(line, color="white", alpha=0.3, linewidth=0.4)
                axes[1, 0].axvline(line, color="white", alpha=0.3, linewidth=0.4)
            for k, (arm, label) in enumerate(names.items(), start=1):
                heat = models[arm][position[image_id]]
                g = maps[arm][c]["detector"]["group"]
                cg = maps[arm][c]["contrast"]["groups"][0]
                axes[0, k].imshow(crop)
                artist = axes[0, k].imshow(
                    heat[:, g].reshape(14, 14),
                    extent=(0, 224, 224, 0),
                    vmin=0,
                    vmax=shared_scale,
                    cmap="magma",
                    alpha=0.65,
                    interpolation="nearest",
                )
                axes[0, k].set_title(
                    f"{label}\nDetector group {g}; capture group {cg}", fontsize=10
                )
                # Dominant native-code group; argmax uses the first index on ties, inactive patches show a dash.
                winners = heat.argmax(1).reshape(14, 14)
                strength = heat.max(1).reshape(14, 14)
                axes[1, k].imshow(crop, extent=(0, 14, 14, 0))
                axes[1, k].set_title(
                    "Highest-activation group ID per patch", fontsize=9
                )
                for y in range(14):
                    for x in range(14):
                        axes[1, k].text(
                            x + 0.5,
                            y + 0.5,
                            str(int(winners[y, x])) if strength[y, x] > 1e-6 else "–",
                            ha="center",
                            va="center",
                            fontsize=5,
                            color="white",
                            bbox=dict(
                                facecolor="black", alpha=0.5, pad=0.2, edgecolor="none"
                            ),
                        )
                known = test["masks"][i]["crowd"] <= cfg["negative_fraction"]
                foreground = (mask.reshape(-1) >= cfg["positive_fraction"]) & known
                bg = (mask.reshape(-1) <= cfg["negative_fraction"]) & known
                energy = heat[foreground].mean(0)
                top = np.argsort(-energy)[:5]
                for rank, h in enumerate(top, 1):
                    rows.append(
                        dict(
                            branch=branch,
                            arm=arm,
                            category=category,
                            image_id=image_id,
                            detector_group=g,
                            capture_group=cg,
                            object_region_rank=rank,
                            group_index=int(h),
                            mean_object_activation=float(energy[h]),
                            mean_other_activation=float(heat[bg, h].mean()),
                            common_color_max=shared_scale,
                            saturated_patch_fraction=float(
                                (heat[:, g] > shared_scale).mean()
                            ),
                            interpretation="Region ranking is descriptive test-image visualization only; it does not choose the evaluated group or train the model. Group indices are local to each model.",
                        )
                    )
            for ax in axes.flat:
                ax.axis("off")
            fig.suptitle(
                f"{branch} · {category} · fixed example; same heat color scale across arms",
                fontsize=12,
            )
            fig.subplots_adjust(wspace=0.04, hspace=0.18, top=0.9, right=0.87)
            color_axis = fig.add_axes([0.9, 0.56, 0.012, 0.29])
            fig.colorbar(artist, cax=color_axis, label="Native coefficient-vector norm")
            name = f"{branch}_c{c}_im{image_id}.png"
            fig.savefig(folder / name, dpi=140, bbox_inches="tight")
            plt.close(fig)
            figures.append(
                dict(branch=branch, category=category, path=name, image_id=image_id)
            )
        artifacts = [r["path"] for r in figures[figure_start:]] + [
            branch + "_all_group_heat.npz"
        ]
        master.save(
            "gallery_done_" + branch,
            dict(
                rows=rows[row_start:],
                figures=figures[figure_start:],
                files={name: sha_file(folder / name) for name in artifacts},
            ),
        )
        master.event("heatmap_branch_complete", branch=branch, examples=len(examples))
    with (folder / "region_group_indices.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cards = "".join(
        f'<h2>{html.escape(r["branch"])} · {html.escape(r["category"])}</h2><img loading="lazy" src="{r["path"]}" alt="Group activation overlays">'
        for r in figures
    )
    (folder / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>AGF group heatmaps</title><style>body{max-width:1400px;margin:32px auto;font:16px system-ui;padding:20px}img{width:100%}h2{margin-top:50px}</style><h1>Where each group activates</h1><p>Examples were fixed before training. Detector/capture groups were chosen on calibration images. Group indices are zero-based and local to each model. The lower panels show the strongest native group at each patch, not ground-truth concept labels. Shared calibration-based color scales avoid per-image rescaling; dark areas may be inactive. Brightness is activation, not proof of causal importance.</p><a href="region_group_indices.csv">Download region-to-group table</a>'
        + cards,
        encoding="utf-8",
    )
    master.write(
        "heatmap_manifest.json",
        dict(
            figures=figures,
            group_indices="heatmaps/region_group_indices.csv",
            raw_maps="heatmaps/*_all_group_heat.npz",
        ),
    )
    return folder / "index.html"
