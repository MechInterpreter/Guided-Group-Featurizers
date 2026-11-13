"""Verified public-data inputs and local journals for historical study recipes."""

import json
import os
from pathlib import Path
import sys
from functools import lru_cache


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return __import__("hashlib").file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".pending")

    def convert(item):
        if isinstance(item, Path):
            return str(item)
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError("Unsupported receipt value: " + type(item).__name__)

    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


@lru_cache(maxsize=None)
def verify_profile(code, profile):
    source = Path(code) / "sources" / profile
    if source.resolve().parent != (Path(code) / "sources").resolve():
        raise ValueError("Invalid historical source profile")
    for name, expected in read(source / "FILE_HASHES.json").items():
        if sha(artifact(source, name)) != expected:
            raise ValueError("Historical implementation changed: " + name)
    return source


def activate(code, profile="voc_slics_compute"):
    code = Path(code).resolve()
    source = code / "sources" / profile
    if not (source / "FILE_HASHES.json").is_file():
        raise FileNotFoundError("Unknown historical source profile: " + profile)
    verify_profile(code, profile)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    sys.path[:0] = [str(source), str(source / "core"), str(code / "training")]
    return source


def artifact(root, name):
    root = Path(root).resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact path leaves its run directory")
    return path


def separate(root, *inputs):
    root = Path(root).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if root == source or root in source.parents or source in root.parents:
            raise ValueError("Choose a study output directory separate from its inputs")
    return root


def own(root, code, inputs, driver, kind):
    """Reserve a new output directory before preparing nested inputs."""
    root = separate(root, code, *inputs)
    payload = dict(
        schema="ggf-fresh-study-owner-v1",
        kind=kind,
        driver_sha256=sha(driver),
        input_loader_sha256=sha(__file__),
        inputs=[str(Path(path).resolve()) for path in inputs],
    )
    marker = root / "OWNER.json"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("Use an empty output directory or resume its own fresh study")
    if marker.exists() and read(marker) != payload:
        raise ValueError("Study source or dependency paths changed")
    root.mkdir(parents=True, exist_ok=True)
    write(marker, payload)
    return payload


def audit_ids(code, cohort="controls"):
    code = Path(code)
    spec = read(code / "bootstrap/SPECIFICATION.json")
    count = 1500
    if cohort == "controls":
        references = [
            read(code / "evidence" / folder / "test_identity.json")["ids"]
            for folder in ("cap040_test", "pilot4096_test", "continuation5")
        ]
        ids = references[0]
        if (
            any(other != ids for other in references[1:])
            or ids != spec["splits"]["audit"]
        ):
            raise ValueError("Historical control audit allocations disagree")
    elif cohort == "supervised-five":
        ids = read(code / "evidence/supervised_five_seed/fresh_holdout_plan.json")[
            "splits"
        ]["test"]
    elif cohort == "legacy-mapping":
        ids = read(code / "bootstrap/LEGACY_ALLOCATIONS.json")["splits"]["mapping"]
        count = 400
    else:
        raise ValueError("Unknown historical audit cohort")
    training = {
        i for split in ("train", "select", "confirm") for i in spec["splits"][split]
    }
    if len(ids) != count or len(set(ids)) != count or training.intersection(ids):
        raise ValueError("Invalid historical audit size or split overlap")
    return ids


def numerical_environment(code, reference="current_control", require_cuda=True):
    import torch
    from coco_ggf.matched_baselines import versions

    expected = read(Path(code) / "evidence" / reference / "manifest.json")["config"][
        "versions"
    ]
    if versions() != expected:
        raise RuntimeError(
            "Install the recorded numerical versions: " + json.dumps(expected)
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for historical fitting and inference"
        )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return expected


def development_inputs(code, source):
    import numpy as np
    import torch
    from coco_baselines.storage import identity

    code, source = Path(code), Path(source)
    marker = read(source / "FRESH_RUN.json")
    if marker["specification_sha256"] != sha(code / "bootstrap/SPECIFICATION.json"):
        raise ValueError("Fresh input specification changed")
    if marker["implementation_sha256"] != sha(code / "bootstrap/fresh_matched75.py"):
        raise ValueError("Fresh feature extractor changed")
    training = {
        p.relative_to(code).as_posix(): sha(p)
        for p in sorted((code / "training").rglob("*.py"))
    }
    if marker["training_source_sha256"] != identity(training):
        raise ValueError("Fresh training-source identity changed")
    spec = read(code / "bootstrap/SPECIFICATION.json")
    features = read(source / "features/FEATURES.json")
    if features["run"] != marker:
        raise ValueError("Feature lineage differs from its run receipt")
    folder = source / "features"
    normalizer = artifact(folder, features["normalization"]["path"])
    if sha(normalizer) != features["normalization"]["sha256"]:
        raise ValueError("Normalization checksum differs")
    norm = torch.load(normalizer, map_location="cpu", weights_only=False)
    packs = {}
    for split in ("train", "select", "confirm"):
        row = features["packs"][split]
        if (
            sha(artifact(folder, row["array"])) != row["array_sha256"]
            or sha(artifact(folder, row["masks"])) != row["masks_sha256"]
        ):
            raise ValueError("Feature checksum differs: " + split)
    try:
        for split in ("train", "select", "confirm"):
            row = features["packs"][split]
            array, mask_file = (
                artifact(folder, row["array"]),
                artifact(folder, row["masks"]),
            )
            masks = torch.load(mask_file, map_location="cpu", weights_only=False)
            if (
                masks["ids"] != spec["splits"][split]
                or row["ids"] != masks["ids"]
                or len(masks["masks"]) != len(masks["ids"])
            ):
                raise ValueError("Feature allocation differs: " + split)
            x = np.load(array, mmap_mode="r")
            packs[split] = dict(
                x=x, ids=row["ids"], masks=masks["masks"], identity=row["identity"]
            )
            if (
                list(x.shape) != [len(spec["splits"][split]), 196, 768]
                or x.dtype != np.float32
            ):
                raise ValueError("Feature dimensions differ: " + split)
    except BaseException:
        for pack in packs.values():
            pack["x"]._mmap.close()
        raise
    return spec, marker, norm, packs


def journal(root, code, feature_root, profile, driver, recipe, versions, norm, packs):
    from coco_adaptation.inputs import JournalStore
    from coco_baselines.storage import identity
    from coco_ggf.drive_recovery import install_drive_recovery

    root = separate(root, code, feature_root)
    source = Path(code) / "sources" / profile
    payload = dict(
        schema="ggf-public-input-historical-study-v1",
        protocol=recipe,
        versions=versions,
        feature_run=read(Path(feature_root) / "FRESH_RUN.json"),
        training_identity=packs["train"]["identity"],
        selection_identity=packs["select"]["identity"],
        confirmation_identity=packs["confirm"]["identity"],
        normalization_sha=identity(norm),
        driver_sha256=sha(driver),
        input_loader_sha256=sha(__file__),
        source_sha=sha(driver),
        historical_sources={
            p.relative_to(source).as_posix(): sha(p)
            for p in sorted(source.rglob("*.py"))
        },
    )
    marker = root / "FRESH_STUDY.json"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError(
            "Use an empty study directory or resume its existing fresh run"
        )
    if marker.exists() and read(marker) != payload:
        raise ValueError("Study source, inputs or recipe changed")
    root.mkdir(parents=True, exist_ok=True)
    write(marker, payload)
    local = root / "scratch"
    local.mkdir(exist_ok=True)
    store = JournalStore(root / "models", payload, local / "rescue", False)
    install_drive_recovery(store)
    store.write("protocol.json", payload)
    return store, local


def audit_pack(ctx, code, root, cohort, frozen):
    """Extract a declared audit cohort after the caller verifies every endpoint."""
    import numpy as np
    import torch
    from PIL import Image
    from coco_baselines import data
    from coco_baselines.storage import array_sha, identity
    from coco_stratified import dataset

    if not frozen.get("complete") or not frozen.get("models"):
        raise ValueError("Freeze the complete model catalog before audit extraction")
    ids = audit_ids(code, cohort)
    spec = read(Path(code) / "bootstrap/SPECIFICATION.json")
    receipt = identity(
        dict(cohort=cohort, ids=ids, normalization=identity(ctx["norm"]), models=frozen)
    )
    folder = Path(root) / "audit" / receipt
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / "COMPLETE.json"
    if marker.exists():
        saved = read(marker)
        if (
            saved["identity"] != receipt
            or sha(folder / "x.npy") != saved["array_sha256"]
            or sha(folder / "masks.pt") != saved["masks_sha256"]
        ):
            raise ValueError("Audit cache identity or checksum differs")
        masks = torch.load(folder / "masks.pt", map_location="cpu", weights_only=False)
        if masks["ids"] != ids:
            raise ValueError("Audit mask allocation differs")
        return dict(
            x=np.load(folder / "x.npy", mmap_mode="r"),
            ids=ids,
            masks=masks["masks"],
            identity=receipt,
        )
    coco, _, annotation_sha = dataset.annotations(
        Path(root) / "public_inputs", ctx["local"] / "annotations"
    )
    if annotation_sha != spec["annotation_sha256"]:
        raise ValueError("COCO annotation checksum differs")
    processor = backbone = None
    target = np.lib.format.open_memmap(
        folder / "x.pending.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(ids), 196, 768),
    )
    masks = []
    try:
        for lo in range(0, len(ids), 8):
            chosen = ids[lo : lo + 8]
            key = "public_audit_" + receipt + "_" + str(lo)
            part = ctx["store"].load(key)
            if part is None:
                if backbone is None:
                    processor, backbone = dataset.backbone(
                        spec, ctx["device"], lambda: os.environ.get("HF_TOKEN")
                    )
                crops, mm = [], []
                for image_id in chosen:
                    info = coco.imgs[image_id]
                    path = data.download(
                        data.COCO_BUCKET + "train2017/" + info["file_name"],
                        Path(root) / "public_inputs/images" / info["file_name"],
                    )
                    geometry = data.geometry(info["width"], info["height"])
                    with Image.open(path) as image:
                        crops.append(
                            data.transform_image(image.convert("RGB"), geometry)
                        )
                    mm.append(
                        data.masks_for_image(
                            coco, image_id, geometry, sorted(coco.cats)
                        )
                    )
                inputs = processor(
                    images=crops,
                    do_resize=False,
                    do_center_crop=False,
                    return_tensors="pt",
                ).to(ctx["device"])
                with torch.inference_mode():
                    raw = (
                        backbone(**inputs).last_hidden_state[:, 5:].float().cpu().half()
                    )
                part = dict(ids=chosen, x=raw, masks=mm)
                ctx["store"].save(key, part)
            if (
                part["ids"] != chosen
                or tuple(part["x"].shape) != (len(chosen), 196, 768)
                or not torch.isfinite(part["x"]).all()
            ):
                raise ValueError("Invalid audit feature shard")
            target[lo : lo + len(chosen)] = data.normalize(
                part["x"].float().numpy(), ctx["norm"]
            )
            masks.extend(part["masks"])
            print(
                "Historical audit", cohort, lo + len(chosen), "/", len(ids), flush=True
            )
    finally:
        del backbone, processor
    target.flush()
    tensor_sha = array_sha(target)
    del target
    os.replace(folder / "x.pending.npy", folder / "x.npy")
    torch.save(dict(ids=ids, masks=masks), folder / "masks.pt")
    write(
        marker,
        dict(
            complete=True,
            identity=receipt,
            cohort=cohort,
            ids=ids,
            normalized_sha=tensor_sha,
            array_sha256=sha(folder / "x.npy"),
            masks_sha256=sha(folder / "masks.pt"),
        ),
    )
    return dict(
        x=np.load(folder / "x.npy", mmap_mode="r"),
        ids=ids,
        masks=masks,
        identity=receipt,
    )


def coco_context(root, source, code, profile, driver, recipe):
    import copy

    activate(code, profile)
    from coco_ggf.width_balance import settings as balance_settings
    from scale_plan import settings

    versions = numerical_environment(code)
    spec, marker, norm, packs = development_inputs(code, source)
    store, local = journal(
        root, code, source, profile, driver, recipe, versions, norm, packs
    )
    cfg = dict(
        spec["fit_config"],
        batch=512,
        audit_image_batch=1,
        bootstrap=1000,
        fresh_test_seed=126101,
        preview_images=0,
    )
    provenance = dict(
        categories=spec["categories"],
        splits=spec["splits"],
        backbone=spec["backbone"],
        backbone_weights=spec["backbone_weights"],
        annotation_sha=spec["annotation_sha256"],
    )
    return dict(
        store=store,
        local=local,
        cfg=cfg,
        norm=norm,
        packs=packs,
        provenance=provenance,
        fresh_provenance=provenance,
        rule=settings(),
        core_rule=settings(),
        balance_recipe=copy.deepcopy(balance_settings()["balance"]),
        device="cuda",
        stamps=[],
        checkpoint_stamps=[],
        source_stamps=[],
        sources={},
        _pairing_proofs={},
        active_key=None,
        options=dict(
            output=str(root / "models"),
            local=str(local),
            wrapper_sha=sha(driver),
            source_sha=sha(driver),
            source_output=str(root / "public_inputs/parent"),
            concept_source=str(code / "training/concept_sae_native.py"),
        ),
    )


class InputView:
    """Write a study's own journal and read explicitly declared fresh dependencies."""

    def __init__(self, job, dependencies=None):
        self.job = job
        self.dependencies = dependencies or {}

    def __getattr__(self, name):
        return getattr(self.job, name)

    def load(self, key):
        value = self.job.load(key)
        if value is None and key in self.dependencies:
            value = self.dependencies[key].load(key)
        return value


def open_study(root, local):
    from coco_baselines.storage import identity
    from coco_coordinated.run import FrozenStore

    config = read(Path(root) / "FRESH_STUDY.json")
    run = identity(config)
    return FrozenStore(Path(root) / "models/checkpoints" / run[:20], Path(local))
