"""Collect and score native responses from a new matched-75 training run."""

import argparse
import os
import sys
from pathlib import Path


def load_banks(root, evaluation):
    from fresh_matched75 import REGIMES

    rows = []
    receipts = {}
    for regime in REGIMES:
        receipt = root / "training" / regime / "BANKS.json"
        manifest = evaluation.read(receipt)
        receipts[regime] = evaluation.sha(receipt)
        for row in manifest["banks"]:
            fold = row["fold"]
            folds = (
                [0, 1]
                if regime == "category_split" and row["kind"] == "vanilla"
                else [fold]
            )
            bank_id = f"{regime}_{row['kind']}_s{row['seed']}_f{fold}"
            rows.append(
                dict(
                    row, bank_id=bank_id, capacity_matched=True, evaluation_folds=folds
                )
            )

    return rows, receipts


def collect(root, code, chunk):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    import collect_evaluation as evaluation
    from fresh_matched75 import RunLease, load_feature_context
    from score_evaluation import numeric

    if not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA runtime for native response collection")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = root / "evaluation"
    with RunLease(root):
        ctx = load_feature_context(root, code, "category_split", include_audit=True)
        rows, receipts = load_banks(root, evaluation)
        out.mkdir(exist_ok=True)
        packs = {"calibration": ctx["packs"]["confirm"], "audit": ctx["packs"]["audit"]}
        ids = {role: pack["ids"] for role, pack in packs.items()}
        from dashboard import validate_splits

        validate_splits(ids["calibration"], ids["audit"])
        training_ids = set(ctx["packs"]["train"]["ids"]) | set(
            ctx["packs"]["select"]["ids"]
        )
        if training_ids.intersection(ids["calibration"] + ids["audit"]):
            raise ValueError("Evaluation IDs overlap fitting or development IDs")
        cats = ctx["eligible"]
        definitions = code / "training" / "TARGET_DEFINITIONS.json"
        targets = evaluation.read(definitions)
        if [row["category_index"] for row in targets["categories"]] != cats:
            raise ValueError("Target category order differs from the paper cohort")
        records = {}
        for role, pack in packs.items():
            for image_id in pack["ids"]:
                records[str(image_id)] = dict(ctx["records"][str(image_id)], role=role)
        protocol = evaluation.read(code / "training" / "PROTOCOL.json")
        evaluation.save(out / "PROTOCOL.json", protocol)
        evaluation.save(out / "TARGET_DEFINITIONS.json", targets)
        evaluation.save(
            out / "ENDPOINT_LOCK.json", dict(banks=rows, training_receipts=receipts)
        )
        evaluation.save(
            out / "INPUTS.json",
            dict(
                schema="fresh-matched75-input-v1",
                ids=ids,
                categories=cats,
                records=records,
                folds=ctx["folds"],
                normalization_identity=evaluation.identity(ctx["norm"]),
                feature_identities={
                    role: pack["identity"] for role, pack in packs.items()
                },
                definitions_sha256=evaluation.sha(out / "TARGET_DEFINITIONS.json"),
            ),
        )
        evaluation.save_labels(out, packs, cats)
        completed = []
        for row in rows:
            completed.append(
                evaluation.collect_bank(
                    out, row, packs, cats, out / "scratch", chunk=chunk
                )
            )
        total = np.zeros(768)
        square = 0.0
        patches = 0
        for image in packs["audit"]["x"]:
            features = np.asarray(image, np.float64)
            total += features.sum(axis=0)
            square += np.square(features).sum()
            patches += len(features)
        evaluation.save(
            out / "COLLECTION_COMPLETE.json",
            dict(
                complete=True,
                scope="fresh_public_data_matched75",
                training_seeds=list(range(21, 28)),
                banks=completed,
                inputs_sha256=evaluation.sha(out / "INPUTS.json"),
                audit_values=patches * 768,
                audit_total_variation=float(square - np.square(total).sum() / patches),
                semantic_scores_status="pending_new_blinded_reviews",
            ),
        )
        numeric(out)


def package(root):
    from eval_common import cache_views, read, save, sha
    from prepare_evaluation_reviews import (
        generic_reviews,
        presence_pack,
        restore_images,
        verify_pack,
    )

    out = root / "evaluation"
    receipt = out / "COLLECTION_COMPLETE.json"
    result = read(receipt)
    if not result["complete"] or result["scope"] != "fresh_public_data_matched75":
        raise ValueError("Complete the fresh native collection first")
    portable = [cache_views(out, bank) for bank in result["banks"]]
    save(
        out / "PORTABLE_LOCK.json",
        dict(complete=True, banks=portable, collection_sha256=sha(receipt)),
    )
    restore_images(out)
    pair = out / "reviews/coherence"
    if (pair / "REVIEW_LOCK.json").exists():
        verify_pack(pair)
        if read(pair / "PRIVATE.json")["collection_sha256"] != sha(receipt):
            raise ValueError("Frozen review collection changed")
    else:
        generic_reviews(out)
    presence = out / "reviews/presence"
    if not (presence / "REVIEW_LOCK.json").exists():
        presence_pack(out)
    else:
        lock = read(presence / "REVIEW_LOCK.json")
        for name in ["PUBLIC", "PRIVATE"]:
            if sha(presence / (name + ".json")) != lock[name.lower() + "_sha256"]:
                raise ValueError("Presence review allocation changed")
        for name, expected in lock["image_sha256"].items():
            if sha(presence / name) != expected:
                raise ValueError("Presence crop changed")
    from fresh_reviews import prepare

    prepare(root, Path(__file__).resolve().parents[1])
    save(
        out / "PACKAGING_COMPLETE.json",
        dict(
            complete=True,
            collection_sha256=sha(receipt),
            pair_lock_sha256=sha(pair / "REVIEW_LOCK.json"),
            whole_lock_sha256=sha(out / "reviews/whole/REVIEW_LOCK.json"),
            presence_lock_sha256=sha(presence / "REVIEW_LOCK.json"),
        ),
    )


def presence_stage(root, phase):
    """Lock crop-presence labels and score concentration.

    Requires calibration/audit labels and independent QA. Group selection and
    threshold calibration precede scoring on the audit split.
    """
    from eval_common import read

    out = root / "evaluation"
    receipt = read(out / "COLLECTION_COMPLETE.json")
    if (
        not receipt.get("complete")
        or receipt.get("scope") != "fresh_public_data_matched75"
    ):
        raise ValueError("Complete the fresh native collection first")
    if phase == "lock-presence":
        from prepare_evaluation_reviews import lock_presence, presence_quality

        lock_presence(out, "calibration")
        lock_presence(out, "audit")
        presence_quality(out)
    elif phase == "select-presence":
        from score_evaluation import select_presence

        select_presence(out)
    elif phase == "concentration":
        from score_evaluation import concentration

        concentration(out)
    else:
        raise ValueError("Unknown presence stage: " + phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "collect",
            "package",
            "lock-presence",
            "select-presence",
            "concentration",
        ],
    )
    parser.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()
    if args.chunk < 1:
        parser.error("--chunk must be positive")
    code = args.code_root.resolve()
    sys.path[:0] = [str(code / "training"), str(code / "training/core")]
    if args.phase == "collect":
        collect(args.root.resolve(), code, args.chunk)
    elif args.phase == "package":
        package(args.root.resolve())
    else:
        presence_stage(args.root.resolve(), args.phase)


if __name__ == "__main__":
    main()
