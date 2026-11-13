"""Reuse exact bank-independent crop labels, then score new concentration curves."""

import argparse, json, shutil
from pathlib import Path
import hashlib
import numpy as np
from PIL import Image
from eval_common import read, sha, save
from prepare_evaluation_reviews import lock_presence, presence_quality
from score_evaluation import select_presence, concentration

HERE = Path(__file__).resolve().parent


def reuse_exact_images(root, drive):
    """Prefer already reviewed exact PNGs; never redraw a different crop."""
    root, drive = Path(root), Path(drive)
    inputs = read(root / "INPUTS.json")
    cases = read(HERE / "prior_presence/reviews/presence/PRIVATE.json")["cases"]
    sources = [
        drive / "GGF_all75_evaluation_20261027",
        drive / "GGF_all75_seed21_pilot_evaluation_20261027",
    ]
    reused = 0
    for case in cases:
        if case["status"] != "ready":
            continue
        iid = case["image_id"]
        target = root / "images" / f"{iid}.png"
        if target.exists():
            if sha(target) != case["crop_sha256"]:
                raise ValueError("New crop differs from reviewed reference")
            continue
        candidates = [
            p / rel
            for p in sources
            for rel in [
                f"images/{iid}.png",
                f"reviews/presence/images/{case['case_id']}.png",
            ]
        ]
        exact = next(
            (p for p in candidates if p.is_file() and sha(p) == case["crop_sha256"]),
            None,
        )
        if exact is None:
            continue  # Original hash-checked JPEG restoration remains available.
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(exact, target)
        record = inputs["records"][str(iid)]
        with Image.open(target) as im:
            pixel_sha = hashlib.sha256(
                np.asarray(im.convert("RGB")).tobytes()
            ).hexdigest()
        save(
            target.with_suffix(".json"),
            dict(
                record,
                status="ready",
                crop_path=str(target.relative_to(root)),
                crop_sha256=sha(target),
                crop_pixel_sha256=pixel_sha,
                pixel_replay_status="Exact previously reviewed PNG bytes reused after SHA verification",
            ),
        )
        reused += 1
    print("Exact existing reviewed PNGs copied:", reused, flush=True)


def reuse_presence(root):
    root = Path(root)
    reference = HERE / "prior_presence"
    manifest = read(reference / "HASHES.json")
    for rel, expected in manifest.items():
        if sha(reference / rel) != expected:
            raise ValueError("Original presence reference changed: " + rel)
    # These include category definitions, case/image correspondence, exact crop
    # hashes and the fixed QA sample. No method response enters a presence label.
    checks = [
        "TARGET_DEFINITIONS.json",
        "reviews/presence/PUBLIC.json",
        "reviews/presence/PRIVATE.json",
        "reviews/presence/REVIEW_LOCK.json",
        "reviews/presence_qa/PUBLIC.json",
    ]
    for rel in checks:
        if sha(root / rel) != sha(reference / rel):
            raise ValueError("Exact crop/definition/QA reuse check failed: " + rel)
    old, new = read(reference / "INPUTS.json"), read(root / "INPUTS.json")
    for field in [
        "ids",
        "categories",
        "definitions_sha256",
        "normalization_identity",
        "feature_identities",
        "folds",
        "records",
        "source_stamp",
    ]:
        if old[field] != new[field]:
            raise ValueError("Input reference changed: " + field)
    for rel in manifest:
        if (
            rel.startswith("reviews/presence/labels/")
            or rel == "reviews/presence_qa/labels.json"
        ):
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha(target) != manifest[rel]:
                raise ValueError("Existing actual labels differ")
            if not target.exists():
                shutil.copyfile(reference / rel, target)
    for role in ["calibration", "audit"]:
        lock_presence(root, role)
    presence_quality(root)
    save(
        root / "ACTUAL_PRESENCE_LABEL_REUSE.json",
        dict(
            complete=True,
            original_source_hashes=manifest,
            reference_not_new_independent_review=True,
            exact_crops_and_definitions_verified=True,
            fixed_QA40_preserved=True,
        ),
    )
    select_presence(root)
    concentration(root)
    print(
        "New calibration-selected concentration/coverage/FPR curves complete; NEW coherence reviews remain pending.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--copy-images-from-drive")
    args = parser.parse_args()
    if args.copy_images_from_drive:
        reuse_exact_images(args.root, args.copy_images_from_drive)
    else:
        reuse_presence(args.root)
