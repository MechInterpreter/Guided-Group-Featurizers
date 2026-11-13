"""Fresh public-data matched75 run, separate from immutable historical replay.

No original checkpoint/cache is required. The frozen implementations supply every
training algorithm; this module supplies public inputs and a new storage lineage.
"""

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import threading

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "SPECIFICATION.json"
REGIMES = ("full_category", "category_split")
EXTRACTION_CACHE_IMPLEMENTATIONS = {
    "758eea0226519c2de821d09848c17f07f66253871bf4a8dee0095187780831c7",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path, value):
    from cd_common import write_json

    write_json(path, value)


def imports(code_root):
    code = Path(code_root).resolve()
    training = code / "training"
    if not (training / "all75_run.py").is_file():
        raise FileNotFoundError("Pass the delivered code/ directory as --code-root")
    sys.path[:0] = [str(training), str(training / "core")]
    return code


def source_identity(code):
    from coco_baselines.storage import identity

    files = {
        str(p.relative_to(code)).replace("\\", "/"): sha(p)
        for p in sorted((code / "training").rglob("*.py"))
    }
    return identity(files)


def specification():
    spec = read(SPEC_PATH)
    if spec["seeds"] != list(range(21, 28)) or len(spec["category_indices"]) != 75:
        raise ValueError("Seven-order matched75 specification changed")
    expected = dict(train=2000, select=400, confirm=400, audit=1500)
    sets = []
    for split, count in expected.items():
        ids = spec["splits"][split]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError("Frozen image allocation is incomplete")
        if any(set(ids).intersection(previous) for previous in sets):
            raise ValueError("Frozen image allocations overlap")
        sets.append(set(ids))
    from cd_common import category_split

    folds = category_split(spec["categories"], spec["category_indices"])
    for actual, original in zip(folds, spec["folds"]):
        if any(actual[k] != original[k] for k in ["fold", "guide", "excluded"]):
            raise ValueError("Frozen category folds changed")
    spec["folds"] = folds
    return spec


def validate_root(root, code):
    root, code = Path(root).resolve(), Path(code).resolve()
    if root == code or root in code.parents or code in root.parents:
        raise ValueError("Fresh run must be outside the delivered source directory")
    if (
        root.exists()
        and any(root.iterdir())
        and not (root / "FRESH_RUN.json").is_file()
    ):
        raise ValueError("Refusing a nonempty directory without a fresh-run identity")
    return root


def prepare(root, code):
    root = validate_root(root, code)
    spec = specification()
    manifest = dict(
        schema="fresh-matched75-run-v1",
        specification_sha256=sha(SPEC_PATH),
        implementation_sha256=sha(__file__),
        training_source_sha256=source_identity(code),
        seeds=spec["seeds"],
        regime_banks=dict(full_category=28, category_split=35),
        historical_weights_reused=False,
        historical_labels_reused=False,
    )
    path = root / "FRESH_RUN.json"
    if path.exists() and read(path) != manifest:
        raise ValueError(
            "Fresh source/specification identity changed; use a new run directory"
        )
    root.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        save(path, manifest)
    return spec, manifest


def environment(require_cuda=True):
    import torch
    from coco_ggf.matched_baselines import versions

    expected = specification()["numerical_versions"]
    actual = versions()
    if actual != expected:
        raise RuntimeError(
            "Install the recorded numerical versions before execution. Expected "
            + json.dumps(expected)
            + "; observed "
            + json.dumps(actual)
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA GPU is required for full extraction/training; no silent CPU fallback"
        )
    torch.set_num_threads(1)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return actual


def required_banks(regime):
    kinds = (
        [(0, "vanilla"), (0, "supervised_vanilla"), (0, "ggf"), (0, "slics")]
        if regime == "full_category"
        else [
            (0, "vanilla"),
            (0, "supervised_vanilla"),
            (0, "ggf"),
            (1, "supervised_vanilla"),
            (1, "ggf"),
        ]
    )
    return {(seed, fold, kind) for seed in range(21, 28) for fold, kind in kinds}


def require_frozen(root, verify_weights=True):
    """No audit feature access until both complete, exact-capacity matrices freeze."""
    from eval_common import endpoint_state

    receipts = {}
    for regime in REGIMES:
        path = Path(root) / "training" / regime / "BANKS.json"
        if not path.is_file():
            raise RuntimeError(
                "Freeze every bank in both regimes before accessing audit features"
            )
        receipt = read(path)
        rows = receipt["banks"]
        keys = {(r["seed"], r["fold"], r["kind"]) for r in rows}
        if (
            not receipt.get("complete")
            or keys != required_banks(regime)
            or len(rows) != len(keys)
        ):
            raise ValueError("Incomplete or duplicate bank matrix: " + regime)
        for row in rows:
            if row["regime"] != regime or (
                row["groups"],
                row["width"],
                row["top_k"],
            ) != (75, 12, None if row["kind"] == "slics" else 32):
                raise ValueError("Frozen capacity or regime differs")
            if verify_weights:
                endpoint_state(row, Path(root) / "scratch/read_frozen")
        receipts[regime] = sha(path)
    return receipts


class RunLease:
    def __init__(self, root):
        from cd_common import Lease

        self.lease = Lease(root)
        self.stop = threading.Event()
        self.errors = []

    def __enter__(self):
        self.lease.acquire()

        def beat():
            while not self.stop.wait(30):
                try:
                    self.lease.beat()
                except Exception as error:
                    self.errors.append(error)
                    os.kill(os.getpid(), signal.SIGTERM)
                    return

        self.thread = threading.Thread(target=beat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=2)
        self.lease.release()


def previous_feature_store(source, root, manifest):
    """Open compatible backbone shards without importing models or labels."""
    from coco_adaptation.inputs import ReadStore
    from coco_baselines.storage import identity

    source = Path(source).resolve()
    if source == root or source in root.parents or root in source.parents:
        raise ValueError("Feature cache and new run must be separate directories")
    original = read(source / "FRESH_RUN.json")
    implementation = original.get("implementation_sha256")
    if implementation not in EXTRACTION_CACHE_IMPLEMENTATIONS | {
        manifest["implementation_sha256"]
    }:
        raise ValueError("Unrecognized feature-cache implementation")
    expected = dict(manifest, implementation_sha256=implementation)
    if original != expected:
        raise ValueError("Feature-cache specification or scientific source differs")
    store = ReadStore(
        source / "features/journal/checkpoints" / identity(original)[:20],
        source / "scratch/features/journal",
    )
    if store.config != original:
        raise ValueError("Feature cache journal does not match its run manifest")
    return store, dict(
        source_run=str(source),
        source_manifest_sha256=sha(source / "FRESH_RUN.json"),
        source_implementation_sha256=implementation,
        scope="Checksum-verified backbone feature shards only; no fits or review labels",
    )


def extract(root, code, audit=False, feature_cache=None):
    import numpy as np
    import torch
    from PIL import Image
    from coco_baselines import data
    from coco_baselines.storage import identity
    from feature_arrays import array_sha, normalization, normalize_to_file
    from coco_stratified import dataset
    from cd_common import make_store

    spec, manifest = prepare(root, code)
    environment()
    receipts = require_frozen(root) if audit else None
    features = root / "features"
    scratch = root / "scratch/features"
    features.mkdir(exist_ok=True)
    coco, annotation, annotation_sha = dataset.annotations(features / "assets", scratch)
    if annotation_sha != spec["annotation_sha256"]:
        raise ValueError("Public COCO annotation bytes differ from frozen allocation")
    if [coco.cats[c] for c in sorted(coco.cats)] != spec["categories"]:
        raise ValueError("COCO category order changed")
    store = make_store(features / "journal", scratch / "journal", manifest, drive=False)
    previous = None
    if feature_cache is not None:
        if audit:
            raise ValueError("Feature-cache import is limited to development splits")
        previous, cache_provenance = previous_feature_store(
            feature_cache, root, manifest
        )
    norm = store.load("normalization")
    if audit and norm is None:
        raise RuntimeError("Extract training inputs before the audit phase")
    index_path = features / "FEATURES.json"
    index = (
        read(index_path)
        if index_path.exists()
        else dict(
            schema="fresh-matched75-features-v1",
            run=manifest,
            annotation_path=str(annotation),
            annotation_sha256=annotation_sha,
            categories=spec["categories"],
            eligible=spec["category_indices"],
            folds=spec["folds"],
            backbone=spec["backbone"],
            backbone_weights=spec["backbone_weights"],
            packs={},
            records={},
        )
    )
    if index["run"] != manifest:
        raise ValueError("Feature lineage changed")
    if previous is not None:
        recorded = index.get("feature_cache_import")
        if recorded and any(recorded[k] != v for k, v in cache_provenance.items()):
            raise ValueError("Feature cache source changed during extraction")
        index.setdefault(
            "feature_cache_import", dict(cache_provenance, imported_shards=[])
        )
    if receipts:
        index["audit_training_receipts"] = receipts
    processor = model = None
    try:
        for split in ["audit"] if audit else ["train", "select", "confirm"]:
            ids = spec["splits"][split]
            raw_path = scratch / (split + "_raw.pending.npy")
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw = np.lib.format.open_memmap(
                raw_path, mode="w+", dtype=np.float32, shape=(len(ids), 196, 768)
            )
            masks, records = [], {}
            for lo in range(0, len(ids), 20):
                chosen = ids[lo : lo + 20]
                key = f"features_{split}_{lo:06d}"
                pack = store.load(key)
                if pack is None and previous is not None:
                    pack = previous.load(key)
                    if pack is not None:
                        if pack["ids"] != chosen or tuple(pack["x"].shape) != (
                            len(chosen),
                            196,
                            768,
                        ):
                            raise ValueError(
                                "Feature-cache shard allocation or shape differs"
                            )
                        store.save(key, pack)
                        index["feature_cache_import"]["imported_shards"].append(key)
                        save(
                            features / "FEATURE_CACHE_IMPORT.json",
                            index["feature_cache_import"],
                        )
                        print(
                            "REUSE FEATURES",
                            split,
                            lo + len(chosen),
                            "/",
                            len(ids),
                            flush=True,
                        )
                if pack is None:
                    if model is None:
                        processor, model = dataset.backbone(
                            spec, "cuda", lambda: os.environ.get("HF_TOKEN")
                        )
                        index["backbone_verification"] = (
                            model._agf_backbone_verification
                        )
                    tensors, mm, geometries, hashes = [], [], [], []
                    for start in range(0, len(chosen), 8):
                        crops = []
                        for image_id in chosen[start : start + 8]:
                            im = coco.imgs[image_id]
                            url = data.COCO_BUCKET + "train2017/" + im["file_name"]
                            path = data.download(
                                url, features / "images" / im["file_name"]
                            )
                            digest = sha(path)
                            expected = spec["expected_evaluation_images"].get(
                                str(image_id)
                            )
                            if expected and digest != expected["original_sha256"]:
                                raise ValueError(
                                    "Public image bytes differ from frozen evaluation image"
                                )
                            geom = data.geometry(im["width"], im["height"])
                            if expected and geom != expected["geometry"]:
                                raise ValueError("Image transform geometry changed")
                            with Image.open(path) as image:
                                if image.size != (im["width"], im["height"]):
                                    raise ValueError(
                                        "Public image size disagrees with annotation"
                                    )
                                crops.append(
                                    data.transform_image(image.convert("RGB"), geom)
                                )
                            geometries.append(geom)
                            hashes.append(digest)
                            mm.append(
                                data.masks_for_image(
                                    coco, image_id, geom, sorted(coco.cats)
                                )
                            )
                        inputs = processor(
                            images=crops,
                            do_resize=False,
                            do_center_crop=False,
                            return_tensors="pt",
                        ).to("cuda")
                        with torch.inference_mode():
                            x = model(**inputs).last_hidden_state[:, 5:].float().cpu()
                        if (
                            tuple(x.shape[1:]) != (196, 768)
                            or not torch.isfinite(x).all()
                        ):
                            raise ValueError(
                                "Unexpected/nonfinite DINOv3 patch features"
                            )
                        tensors.append(x.half())
                    pack = dict(
                        ids=chosen,
                        x=torch.cat(tensors),
                        masks=mm,
                        geometry=geometries,
                        image_hashes=hashes,
                    )
                    store.save(key, pack)
                if pack["ids"] != chosen:
                    raise ValueError("Feature shard order changed")
                raw[lo : lo + len(chosen)] = pack["x"].float().numpy()
                masks.extend(pack["masks"])
                for image_id, geometry, digest in zip(
                    chosen, pack["geometry"], pack["image_hashes"]
                ):
                    records[str(image_id)] = dict(
                        image_id=image_id,
                        geometry=geometry,
                        original_sha256=digest,
                        source_url=data.COCO_BUCKET
                        + "train2017/"
                        + coco.imgs[image_id]["file_name"],
                    )
                print("EXTRACT", split, lo + len(chosen), "/", len(ids), flush=True)
                del pack
            raw.flush()
            if split == "train":
                if norm is None:
                    print("NORMALIZE", split, "training statistics", flush=True)
                    norm = normalization(raw)
                    store.save("normalization", norm)
                elif array_sha(raw) != norm["train_sha"]:
                    raise ValueError(
                        "Training activations changed against frozen normalization"
                    )
            array_path, mask_path = (
                features / f"{split}.npy",
                features / f"{split}_masks.pt",
            )
            tmp = array_path.with_suffix(".pending.npy")
            print("NORMALIZE", split, "writing feature array", flush=True)
            normalize_to_file(raw, norm, tmp)
            del raw
            raw_path.unlink()
            if array_path.exists() and sha(array_path) != sha(tmp):
                raise ValueError("Existing normalized features changed")
            os.replace(tmp, array_path)
            normalized = np.load(array_path, mmap_mode="r")
            mask_payload = dict(ids=ids, masks=masks)
            store.save("masks_" + split, mask_payload)
            # JSON hashes validate serialized bytes; pack identity validates tensors/masks.
            tmp_mask = mask_path.with_suffix(".pending.pt")
            torch.save(mask_payload, tmp_mask)
            os.replace(tmp_mask, mask_path)
            ident = identity(
                dict(
                    ids=ids,
                    normalized_sha=array_sha(normalized),
                    normalization=identity(norm),
                    masks=masks,
                )
            )
            index["packs"][split] = dict(
                ids=ids,
                array=array_path.name,
                masks=mask_path.name,
                array_sha256=sha(array_path),
                masks_sha256=sha(mask_path),
                identity=ident,
                normalized_sha=array_sha(normalized),
                shape=list(normalized.shape),
            )
            index["records"].update(records)
            norm_path = features / "normalization.pt"
            if not norm_path.exists():
                torch.save(norm, norm_path)
            elif identity(
                torch.load(norm_path, map_location="cpu", weights_only=False)
            ) != identity(norm):
                raise ValueError("Serialized normalization differs")
            index["normalization"] = dict(
                path=norm_path.name,
                sha256=sha(norm_path),
                identity=identity(norm),
                historical_identity=spec["historical_identity_comparison"][
                    "normalization"
                ],
                historical_identity_matches=identity(norm)
                == spec["historical_identity_comparison"]["normalization"],
            )
            save(index_path, index)
            del masks, normalized
            gc.collect()
    finally:
        del model, processor
        torch.cuda.empty_cache()
    return index


def load_feature_context(root, code_root, regime, include_audit=False, device="cuda"):
    import numpy as np
    import torch

    code = imports(code_root)
    import all75_run
    import cd_common
    from coco_baselines.storage import identity
    from coco_ggf.matched_baselines import versions
    from coco_ggf.width_balance import settings
    from scale_plan import settings as full_settings

    root = Path(root).resolve()
    spec, manifest = prepare(root, code)
    receipts = require_frozen(root) if include_audit else None
    index = read(root / "features/FEATURES.json")
    if index["run"] != manifest:
        raise ValueError("Feature source identity differs from fresh-run manifest")
    if include_audit and index.get("audit_training_receipts") != receipts:
        raise ValueError("Audit features do not identify both frozen training receipts")
    norm_path = root / "features" / index["normalization"]["path"]
    if sha(norm_path) != index["normalization"]["sha256"]:
        raise ValueError("Normalization bytes changed")
    norm = torch.load(norm_path, map_location="cpu", weights_only=False)
    packs = {}
    for split in ["train", "select", "confirm"] + (["audit"] if include_audit else []):
        rec = index["packs"][split]
        path, masks = root / "features" / rec["array"], root / "features" / rec["masks"]
        if sha(path) != rec["array_sha256"] or sha(masks) != rec["masks_sha256"]:
            raise ValueError("Feature or mask artifact changed: " + split)
        x = np.load(path, mmap_mode="r")
        mask_pack = torch.load(masks, map_location="cpu", weights_only=False)
        if (
            list(x.shape) != [len(spec["splits"][split]), 196, 768]
            or mask_pack["ids"] != spec["splits"][split]
            or rec["ids"] != mask_pack["ids"]
        ):
            raise ValueError("Feature/mask shape or allocation mismatch")
        packs[split] = dict(
            x=x, ids=rec["ids"], masks=mask_pack["masks"], identity=rec["identity"]
        )
    if include_audit:
        packs["test"] = packs["audit"]
    all75_run.configure(regime)
    cd_common.SPEC.update(seeds=list(range(21, 28)))
    folds = (
        [dict(fold=0, guide=spec["category_indices"], excluded=[])]
        if regime == "full_category"
        else spec["folds"]
    )
    options = dict(
        output=str(root / "training" / regime),
        local=str(root / "scratch" / regime),
        regime=regime,
        experiment_sha=manifest["implementation_sha256"],
        core_sha=manifest["training_source_sha256"],
        annotation_path=index["annotation_path"],
        paths={},
        concept_source="unused-in-matched75",
    )
    rule = full_settings()
    cfg = dict(
        spec["fit_config"],
        batch=512,
        audit_image_batch=1,
        bootstrap=1000,
        max_audit_groups=4,
        preview_images=0,
    )
    payload = dict(
        fresh=True,
        manifest=manifest,
        regime=regime,
        folds=folds,
        normalization=identity(norm),
        train=packs["train"]["identity"],
        development=packs["select"]["identity"],
        versions=versions(),
    )
    store = cd_common.make_store(
        Path(options["output"]) / "journal",
        Path(options["local"]) / "journal",
        payload,
        drive=False,
    )
    return dict(
        options=options,
        store=store,
        packs=packs,
        cfg=cfg,
        device=device,
        norm=norm,
        provenance=dict(
            categories=spec["categories"], annotation_sha=spec["annotation_sha256"]
        ),
        folds=folds,
        eligible=spec["category_indices"],
        versions=versions(),
        local=Path(options["local"]),
        core_rule=rule,
        rule=rule,
        balance_recipe=copy.deepcopy(settings()["balance"]),
        active_key=None,
        records=index["records"],
        test_ids=spec["splits"]["audit"],
        test_stamp=index["packs"].get("audit"),
        writable_stores=[],
        stamps=[],
        checkpoint_stamps=[],
    )


def fresh_selection(ctx, fold, kind):
    """Same frozen seed21 selection rule, resolved inside this new run only."""
    import cd_train
    from coco_coordinated.run import required

    if kind != "supervised_vanilla":
        raise ValueError("Only supervised Vanilla selects a fixed loss pair")
    store = cd_train.get_store(ctx, fold, 21, kind)
    if not required(store, "frozen")["training_complete"]:
        raise ValueError("Complete seed21 before applying its fixed selection")
    selection = required(store, "fixed_weight_selection")
    if selection["weights"] not in [
        [1.0, 0.05, 0.01],
        [1.0, 0.2, 0.2],
        [1.0, 0.4, 0.4],
    ]:
        raise ValueError("Unexpected seed21 fixed-loss candidate")
    return selection


def full_native_bank(ctx, seed, kind):
    import cd_train
    import cd_common
    import supervised_study as study
    from coco_coordinated.run import required

    store = cd_train.get_store(ctx, 0, seed, kind)
    old = cd_train.frozen_row(store)
    if old is not None:
        return old
    if kind == "ggf":
        warm, targets = study.small_warm(ctx, seed)
        key = f"small75_s{seed}_lr10000_ggf"
        state = study.fit(
            study.small_context(ctx),
            key,
            "released_vanilla",
            75,
            32,
            seed,
            warm["weights"],
            3000,
            3000,
            targets=targets,
            cap=0.6,
            learning_rate=0.001,
            stream=1,
            keep_candidates=True,
            readout_targets=targets,
        )
        if state["updates"] != 3000 or warm["updates"] != 7640:
            raise ValueError("Full GGF endpoint incomplete")
        store.save("warm", warm)
        store.save("continuation", state)
    elif kind == "slics":
        study.native_fit(ctx, seed)
        state = required(ctx["store"], f"native_slics12_s{seed}")
        if state["updates"] != 10640:
            raise ValueError("Native SLiCS endpoint incomplete")
        warm = required(ctx["store"], f"small75_s{seed}_warm")
        final = required(ctx["store"], f"small75_s{seed}_lr10000_ggf")
        if (
            state["phase_batch_shas"]["warm"] != warm["sampled_batches_sha"]
            or state["sampled_batches_sha"] != final["sampled_batches_sha"]
        ):
            raise ValueError(
                "Full GGF and SLiCS saw different training image schedules"
            )
        store.save("native_training", state)
    else:
        raise ValueError("Unknown full-category native fit")
    return cd_common.finish_model(store, state, kind, 0, seed)


def train(root, code, regime):
    import torch
    import all75_run
    import cd_common
    import cd_train
    import seeds24_27_run
    from eval_common import endpoint_state

    prepare(root, code)
    environment()
    ctx = load_feature_context(root, code, regime)
    cd_train.training_context = all75_run.training_context
    seeds24_27_run.frozen_selection = fresh_selection
    if regime == "full_category":
        seeds24_27_run.install_legacy_precision(ctx["store"])
    rows = []
    try:
        for seed in range(21, 28):
            tasks = [(0, "vanilla")] + [
                (fold, kind)
                for fold in range(len(ctx["folds"]))
                for kind in (
                    ["supervised_vanilla"]
                    if regime == "full_category"
                    else ["supervised_vanilla", "ggf"]
                )
            ]
            if regime == "full_category":
                tasks += [(0, "ggf"), (0, "slics")]
            for fold, kind in tasks:
                print("FRESH FIT", regime, seed, fold, kind, flush=True)
                row = (
                    full_native_bank(ctx, seed, kind)
                    if regime == "full_category" and kind in ["ggf", "slics"]
                    else cd_train.fit_bsf(ctx, fold, seed, kind)
                )
                row.update(
                    regime=regime,
                    origin="new",
                    groups=75,
                    width=12,
                    top_k=None if kind == "slics" else 32,
                    fixed_endpoint=True,
                    total_updates=10640,
                    normalization_identity=__import__(
                        "coco_baselines.storage", fromlist=["identity"]
                    ).identity(ctx["norm"]),
                )
                endpoint_state(row, root / "scratch/verify_endpoints")
                if kind != "slics":
                    final = cd_train.get_store(ctx, fold, seed, kind).load(
                        "continuation"
                    )
                    row.update(
                        continuation_batches_sha=final["sampled_batches_sha"],
                        continuation_noise_sha=final["sampled_noise_sha"],
                    )
                rows.append(row)
                same = [r for r in rows if r["seed"] == seed and r["kind"] != "slics"]
                if (
                    len({r["continuation_batches_sha"] for r in same}) != 1
                    or len({r["continuation_noise_sha"] for r in same}) != 1
                ):
                    raise ValueError("Common BSF image/noise schedules differ")
                save(
                    root / "training" / regime / "BANKS_PROGRESS.json",
                    dict(complete=False, regime=regime, banks=rows),
                )
                gc.collect()
                torch.cuda.empty_cache()
        if {(r["seed"], r["fold"], r["kind"]) for r in rows} != required_banks(regime):
            raise ValueError("Fresh bank matrix is incomplete")
        save(
            root / "training" / regime / "BANKS.json",
            dict(
                complete=True,
                regime=regime,
                schema="fresh-matched75-banks-v1",
                seeds=list(range(21, 28)),
                banks=rows,
                feature_manifest_sha256=sha(root / "features/FEATURES.json"),
                source_manifest_sha256=sha(root / "FRESH_RUN.json"),
                historical_weights_reused=False,
                semantic_review_status="New blinded reviews required; no old labels transferred",
            ),
        )
    finally:
        for store in {
            str(s.root): s for s in [ctx["store"], *ctx["writable_stores"]]
        }.values():
            cd_common.flush(store)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=["prepare", "extract", "train", "extract-audit"],
        required=True,
    )
    parser.add_argument("--regime", choices=REGIMES)
    parser.add_argument("--feature-cache", type=Path)
    args = parser.parse_args()
    code = imports(args.code_root)
    root = args.root.resolve()
    if args.phase == "train" and args.regime is None:
        parser.error("--regime is required for training")
    if args.feature_cache is not None and args.phase != "extract":
        parser.error("--feature-cache is supported only for extract")
    prepare(root, code)
    if args.phase == "prepare":
        print("Fresh run prepared; no downloads, fitting or paid calls were made.")
        return

    def interrupt(*_):
        raise KeyboardInterrupt(
            "Stopping at the last committed training/extraction generation"
        )

    signal.signal(signal.SIGTERM, interrupt)
    with RunLease(root):
        if args.phase == "train":
            train(root, code, args.regime)
        else:
            extract(
                root,
                code,
                audit=args.phase == "extract-audit",
                feature_cache=args.feature_cache,
            )


if __name__ == "__main__":
    main()
