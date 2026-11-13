"""Prepare public VOC inputs and fit the recorded 19-group GGF/SLiCS comparison."""

import argparse
from contextlib import ExitStack
import copy
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch


def setup(code):
    profile = code / "sources/voc_slics_compute"
    sys.path[:0] = [str(profile), str(profile / "core")]
    return profile


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fit_configuration(specification):
    return dict(
        copy.deepcopy(specification["fit_config"]),
        batch=512,
        audit_image_batch=1,
        bootstrap=1000,
        preview_images=0,
    )


def prepare_root(root, code):
    root, code = root.resolve(), code.resolve()
    if root == code or root in code.parents or code in root.parents:
        raise ValueError("Use a new output directory outside the source tree")
    specification = read(code / "bootstrap/SPECIFICATION.json")
    profile = code / "sources/voc_slics_compute"
    identity = dict(
        schema="ggf-fresh-voc19-v1",
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_files={
            p.relative_to(profile).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in sorted(profile.rglob("*.py"))
        },
        backbone=specification["backbone"],
        backbone_weights=specification["backbone_weights"],
        groups=19,
        width=12,
        top_k=8,
        seeds=[21, 22, 23],
        GGF_learning_rate=0.001,
        GGF_cap=0.6,
        warm_updates=7640,
        guided_updates=3000,
        SLiCS_updates=10640,
        category_indices=specification["category_indices"],
        fit_config=fit_configuration(specification),
    )
    marker = root / "FRESH_VOC.json"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("Use an empty output directory or an existing fresh VOC run")
    if marker.exists() and read(marker) != identity:
        raise ValueError("VOC source or input specification changed")
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        write(marker, identity)
    return specification, identity


def pinned_loader(original, pin):
    def load(name, *args, **kwargs):
        if name != pin["model"]:
            raise ValueError("Unexpected backbone model")
        kwargs.update(revision=pin["revision"], token=os.environ.get("HF_TOKEN"))
        return original(name, *args, **kwargs)

    return load


def context(root, code):
    specification, identity = prepare_root(root, code)
    setup(code)
    import torch
    from transformers import AutoImageProcessor, AutoModel
    from coco_adaptation.inputs import JournalStore
    from coco_baselines.storage import identity as fingerprint
    from coco_ggf import matched_baselines, width_balance
    from coco_stratified import dataset
    from scale_plan import settings
    import voc_training
    import voc_pair

    expected = read(code / "evidence/voc_slics_final/manifest.json")["config"][
        "versions"
    ]
    if matched_baselines.versions() != expected:
        raise RuntimeError(
            "Install the recorded VOC numerical versions: " + json.dumps(expected)
        )
    if not torch.cuda.is_available():
        raise RuntimeError("VOC extraction and fitting require CUDA")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    local = root / "scratch"
    local.mkdir(exist_ok=True)
    source_hash = sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    options = dict(
        output=str(root / "outputs"),
        local=str(local),
        source_output=str(root / "inputs/parent"),
        source_sha=source_hash,
        wrapper_sha=source_hash,
        cache_local=str(root / "inputs"),
    )
    parent_store = JournalStore(
        root / "inputs",
        dict(versions=expected, source=identity),
        local / "inputs",
        False,
    )
    provenance = dict(
        categories=specification["categories"],
        backbone=specification["backbone"],
        backbone_weights=specification["backbone_weights"],
    )
    parent = dict(
        options=options,
        local=local,
        cfg=fit_configuration(specification),
        rule=settings(),
        provenance=provenance,
        fresh_provenance=provenance,
        device="cuda",
        store=parent_store,
        balance_recipe=copy.deepcopy(width_balance.settings()["balance"]),
    )
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                AutoModel,
                "from_pretrained",
                side_effect=pinned_loader(
                    AutoModel.from_pretrained, specification["backbone"]
                ),
            )
        )
        stack.enter_context(
            patch.object(
                AutoImageProcessor,
                "from_pretrained",
                side_effect=pinned_loader(
                    AutoImageProcessor.from_pretrained, specification["backbone"]
                ),
            )
        )
        coco, annotation, digest = dataset.annotations(
            root / "inputs/coco", local / "annotations"
        )
        if digest != specification["annotation_sha256"]:
            raise ValueError("COCO annotation checksum differs")
        result = voc_training.prepare(parent, coco, require_drive=False)
    if len(result["cfg"]["category_indices"]) != 19:
        raise ValueError(
            "VOC category denominator differs from the recorded comparison"
        )
    result["core_rule"] = copy.deepcopy(result["rule"])
    result["source_stamps"] = []
    result["store"].write("training_protocol.json", voc_pair.protocol())
    result["store"].save(
        "protocol_frozen_before_new_fits",
        dict(protocol=voc_pair.protocol(), selection_uses_VOC_validation=False),
    )
    write(
        root / "INPUTS_COMPLETE.json",
        dict(
            complete=True,
            plan_sha256=fingerprint(result["voc_plan"]),
            normalization=fingerprint(result["norm"]),
            backbone=specification["backbone"],
            packs={k: p["identity"] for k, p in result["packs"].items()},
            groups=voc_pair.G,
            top_k=voc_pair.K,
        ),
    )
    return result


def validate_rows(rows):
    if len(rows) != 6 or {(r["seed"], r["arm"]) for r in rows} != {
        (s, m) for s in [21, 22, 23] for m in ["GGF", "SLiCS"]
    }:
        raise ValueError("Complete all six VOC fits before evaluation")
    if any(r["selected_total_updates"] != 10640 or r["groups"] != 19 for r in rows):
        raise ValueError("Incomplete VOC endpoint")


def train(root, ctx):
    import torch
    import voc_pair

    rows = []
    for seed in voc_pair.SEEDS:
        for method in ["GGF", "SLiCS"]:
            key = f"fresh_fitted_{method}_{seed}"
            row = ctx["store"].load(key)
            if row is None:
                torch.cuda.reset_peak_memory_stats()
                started = time.monotonic()
                row = voc_pair.final_row(ctx, seed, method)
                row.update(
                    fit_invocation_seconds=time.monotonic() - started,
                    fit_peak_allocated=torch.cuda.max_memory_allocated(),
                    fit_peak_reserved=torch.cuda.max_memory_reserved(),
                )
                ctx["store"].save(key, row)
            rows.append(row)
    validate_rows(rows)
    receipt = dict(
        complete=True, models=rows, input_identity=read(root / "INPUTS_COMPLETE.json")
    )
    write(root / "FITS_COMPLETE.json", receipt)


def evaluate(root, ctx):
    import voc_pair
    from coco_baselines import training
    from coco_coordinated.run import required

    receipt = read(root / "FITS_COMPLETE.json")
    if not receipt.get("complete"):
        raise ValueError("VOC fits are incomplete")
    validate_rows(receipt["models"])
    if receipt["input_identity"] != read(root / "INPUTS_COMPLETE.json"):
        raise ValueError("VOC input identities changed after fitting")
    results = []
    for row in receipt["models"]:
        row = copy.deepcopy(row)
        state = required(ctx["store"], row["key"])
        if state["updates"] != (10640 if row["arm"] == "SLiCS" else 3000):
            raise ValueError("VOC checkpoint is incomplete")
        weights = {"D": state["D"]} if row["arm"] == "SLiCS" else state["weights"]
        if training.weight_sha(weights) != row["weights_sha"]:
            raise ValueError("VOC fitted weights changed")
        row["evaluation"] = voc_pair.evaluate_state(
            ctx, row, ctx["packs"]["eval"], True
        )
        if row["arm"] == "SLiCS":
            row["evaluation"]["reconstruction"] = voc_pair.joint_evaluation(
                ctx, row, ctx["packs"]["eval"]
            )
        results.append(row)
    voc_pair.save_tables(ctx, results)
    write(
        root / "RESULTS.json",
        dict(complete=True, models=results, input_identity=receipt["input_identity"]),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=["prepare", "train", "evaluate"], required=True
    )
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root = args.root.resolve()
    ctx = context(root, args.code_root.resolve())
    if args.phase == "train":
        train(root, ctx)
    elif args.phase == "evaluate":
        evaluate(root, ctx)


if __name__ == "__main__":
    main()
