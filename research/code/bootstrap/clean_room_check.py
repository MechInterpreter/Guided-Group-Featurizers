"""Check public-input GPU execution without historical caches or checkpoints.

The diagnostic extracts features from two training images and checks a
75-group BSF forward/backward pass.
"""

import argparse
import json
import os
from pathlib import Path
import sys


def destination(root, code):
    root, code = Path(root).resolve(), Path(code).resolve()
    if root == code or root in code.parents or code in root.parents:
        raise ValueError("Use a diagnostic directory outside the source tree")
    if root.exists() and any(root.iterdir()):
        raise ValueError("Use an empty diagnostic directory for a clean-room check")
    return root


def training_images(spec):
    ids = spec["splits"]["train"][:2]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("Two distinct training images are required")
    if set(ids).intersection(spec["splits"]["audit"]):
        raise ValueError("Diagnostic images overlap the closed audit split")
    return ids


def run(root, code, preflight_only=False):
    from fresh_matched75 import environment, imports, sha, specification

    root = destination(root, code)
    code = imports(code)
    # Fail before downloads if the recorded environment or CUDA is unavailable.
    versions = environment()
    spec = specification()
    ids = training_images(spec)
    if preflight_only:
        print("Pinned numerical environment and CUDA preflight passed. No downloads.")
        return

    import torch
    from PIL import Image
    from coco_baselines import data
    from coco_stratified import dataset
    from scale_models import model as bsf_model

    root.mkdir(parents=True, exist_ok=True)
    # Isolate even the model provider's cache from an author's existing cache.
    os.environ["HF_HOME"] = str(root / "public_model_cache")
    os.environ["HF_HUB_CACHE"] = str(root / "public_model_cache/hub")
    from huggingface_hub import constants

    constants.HF_HOME = os.environ["HF_HOME"]
    constants.HF_HUB_CACHE = os.environ["HF_HUB_CACHE"]
    constants.HUGGINGFACE_HUB_CACHE = os.environ["HF_HUB_CACHE"]
    processor, backbone = dataset.backbone(
        spec, "cuda", lambda: os.environ.get("HF_TOKEN")
    )
    verification = backbone._agf_backbone_verification
    crops, sources = [], []
    for image_id in ids:
        name = f"{image_id:012d}.jpg"
        path = data.download(
            data.COCO_BUCKET + "train2017/" + name, root / "images" / name
        )
        with Image.open(path) as image:
            geometry = data.geometry(*image.size)
            crops.append(data.transform_image(image.convert("RGB"), geometry))
        sources.append(dict(image_id=image_id, sha256=sha(path), geometry=geometry))
    inputs = processor(
        images=crops, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to("cuda")
    with torch.inference_mode():
        features = backbone(**inputs).last_hidden_state[:, 5:].float()
    if tuple(features.shape) != (2, 196, 768) or not torch.isfinite(features).all():
        raise ValueError("Invalid DINOv3 patch features")
    del backbone, inputs
    # Clone outside inference mode so autograd can save the tensor for backward.
    x = features.clone().reshape(-1, 768)
    del features
    torch.manual_seed(21)
    bsf = bsf_model("released_vanilla", 75, 12, 32, 768).to("cuda")
    loss, _ = bsf.loss(x, target=x)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite BSF diagnostic loss")
    loss.backward()
    gradients = [p.grad for p in bsf.parameters() if p.requires_grad]
    if not gradients or any(
        g is None or not torch.isfinite(g).all() for g in gradients
    ):
        raise ValueError("Invalid BSF backward pass")
    torch.cuda.synchronize()
    receipt = dict(
        schema="ggf-clean-room-smoke-v1",
        complete=True,
        scope="diagnostic_only_not_a_research_result",
        numerical_versions=versions,
        historical_weights_reused=False,
        historical_feature_caches_reused=False,
        images=sources,
        feature_shape=[2, 196, 768],
        groups=75,
        width=12,
        active_groups=32,
        backbone_pin=spec["backbone"],
        backbone_verified=verification,
        bsf_forward_backward_passed=True,
        full_experiment_reproduced=False,
        new_paid_API_calls=False,
    )
    temporary = root / "SMOKE_COMPLETE.json.tmp"
    temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "SMOKE_COMPLETE.json")
    print("GPU smoke check passed.")
    print(root / "SMOKE_COMPLETE.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "bootstrap"))
    from fresh_matched75 import imports

    imports(args.code_root)
    run(args.root, args.code_root, args.preflight_only)


if __name__ == "__main__":
    main()
