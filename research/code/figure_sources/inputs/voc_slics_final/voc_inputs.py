"""Read the completed VOC cohort and raw features without writing source journals."""

import copy, hashlib, io, json, tarfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from coco_baselines import data
from coco_baselines.storage import identity, sha_file
from coco_coordinated.run import FrozenStore, required
from coco_adaptation.inputs import JournalStore
from coco_ggf.drive_recovery import install_drive_recovery
from coco_ggf import matched_baselines, width_balance
from voc import mask_pair, mapping, MD5


def unique_manifest(root, predicate):
    hits = []
    for p in (Path(root) / "results").glob("*/manifest.json"):
        m = json.loads(p.read_text())
        if identity(m["config"]) != m["run_id"]:
            raise ValueError("Invalid manifest: " + str(p))
        if predicate(m["config"]):
            hits.append((p, m))
    if len(hits) != 1:
        raise ValueError(
            "Expected one compatible saved run in "
            + str(root)
            + "; found "
            + str(len(hits))
        )
    return hits[0]


def prepare(o, require_drive=True):
    local = Path(o["local"])
    local.mkdir(parents=True, exist_ok=True)
    root = Path(o["saved_root"]) / "voc_training"
    p, manifest = unique_manifest(
        root, lambda c: c.get("study") == "ggf_VOC_training_v1"
    )
    source = FrozenStore(
        root / "checkpoints" / manifest["run_id"][:20], local / "read_voc"
    )
    plan = source.config["plan"]
    norm = required(source, "normalization")
    if source.config["versions"] != matched_baselines.versions():
        raise ValueError("Saved VOC numerical versions differ")
    cp, cap = unique_manifest(
        Path(o["drive"]) / "AGF_COCO_capacity_sweep_v1",
        lambda c: (
            identity(c)
            == "2921b057f824362ab83e76e7fb35645f94c376fadf8b86f4f22d1a2311105b9e"
        ),
    )
    cfg = copy.deepcopy(cap["config"]["fit_config"])
    cfg.update(
        category_indices=plan["category_indices"],
        bootstrap=1000,
        audit_image_batch=1,
        preview_images=0,
        batch=512,
    )
    provenance = json.loads(
        (
            Path(o["drive"])
            / "AGF_COCO_stratified_confirmation_v1/results"
            / cap["config"]["activation_run"][:20]
            / "provenance.json"
        ).read_text()
    )
    categories = provenance["categories"]
    catmap = mapping(categories)
    if len(plan["category_indices"]) != 19:
        raise ValueError("Expected the existing 19-category VOC denominator")
    archives = [
        Path(o["saved_root"]) / "input_assets/voc2012/VOCtrainval_11-May-2012.tar",
        Path(o["drive"])
        / "GGF_COCO_1024_supervised_VOC_v1/input_assets/voc2012/VOCtrainval_11-May-2012.tar",
    ]
    found = [a for a in archives if a.is_file()]
    if not found:
        raise FileNotFoundError(
            "Restore the existing official VOC archive; no silent dataset replacement"
        )
    archive = found[0]
    md5 = hashlib.md5()
    with archive.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024**2), b""):
            md5.update(part)
    if md5.hexdigest() != MD5:
        raise ValueError("VOC archive checksum changed")
    masks = []
    with tarfile.open(archive) as tar:
        for r in plan["records"]:
            arrays = []
            for folder, key in [
                ("SegmentationClass", "class_sha"),
                ("SegmentationObject", "object_sha"),
            ]:
                name = "VOCdevkit/VOC2012/" + folder + "/" + r["image"] + ".png"
                m = tar.getmember(name)
                if not m.isfile() or m.size > 30 * 1024**2:
                    raise ValueError("Invalid VOC mask archive member")
                a = np.array(Image.open(io.BytesIO(tar.extractfile(m).read())))
                if hashlib.sha256(a.tobytes()).hexdigest() != r[key]:
                    raise ValueError("VOC mask differs from saved plan")
                arrays.append(a)
            masks.append(mask_pair(*arrays, r["geometry"], catmap, r["id"]))
    packs = {}
    stamp = identity(plan)
    for split in ["train", "select", "eval"]:
        idx = [
            i
            for i, r in enumerate(plan["records"])
            if plan["partitions"][r["image"]] == split
        ]
        path = local / (split + ".npy")
        out = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(len(idx), 196, 768)
        )
        loc = {i: j for j, i in enumerate(idx)}
        for lo in range(0, len(plan["records"]), 16):
            rows = [
                i for i in range(lo, min(lo + 16, len(plan["records"]))) if i in loc
            ]
            if not rows:
                continue
            part = required(source, f"voc_raw_{lo:05d}")
            rr = plan["records"][lo : lo + 16]
            if part["plan_sha"] != stamp or part["ids"] != [r["id"] for r in rr]:
                raise ValueError("VOC raw shard lineage changed")
            for i in rows:
                out[loc[i]] = data.normalize(
                    np.asarray(part["x"][i - lo], np.float32)[None], norm
                )[0]
        out.flush()
        del out
        packs[split] = dict(
            x=np.load(path, mmap_mode="r"),
            ids=[plan["records"][i]["id"] for i in idx],
            masks=[masks[i] for i in idx],
            identity=identity(dict(plan=stamp, split=split, norm=identity(norm))),
        )
        print("VOC cached split restored", split, len(idx), flush=True)
    cfg["category_indices"] = plan["category_indices"]
    from voc_pair import protocol

    payload = dict(
        study="ggf_voc_slics_category_matched_compute_v1",
        protocol=protocol(),
        source_run=source.run_id,
        versions=matched_baselines.versions(),
        source_sha=o["extension_sha"],
        train=packs["train"]["identity"],
        development=packs["select"]["identity"],
        evaluation=packs["eval"]["identity"],
        normalization=identity(norm),
    )
    s = JournalStore(o["output"], payload, local / "rescue", require_drive)
    install_drive_recovery(s)
    s.write("protocol.json", payload)
    stamps = [dict(path=str(q), sha=sha_file(q)) for q in [p, cp]]
    return dict(
        options=o,
        store=s,
        source=source,
        local=local,
        cfg=cfg,
        packs=packs,
        norm=norm,
        provenance=provenance,
        rule=copy.deepcopy(source.config["rule"]),
        core_rule=copy.deepcopy(source.config["rule"]),
        balance_recipe=copy.deepcopy(width_balance.settings()["balance"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        source_stamps=stamps,
    )
