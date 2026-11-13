"""Minimum-work integration checks using the paper's implementation functions."""

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import sys

from command_runner import run_command


STAGES = (
    "public",
    "voc_inputs",
    "fits",
    "native",
    "concept",
    "analysis",
    "presence",
    "synthetic",
    "historical",
    "compute",
    "cards",
    "figures",
)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    temp.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temp.replace(path)


def setup(root, code, device):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    root, code = Path(root).resolve(), Path(code).resolve()
    if root == code or root in code.parents or code in root.parents:
        raise ValueError("Use a quick-check directory outside the source tree")
    versions = {}
    for package in (
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "scikit-learn",
        "transformers",
        "Pillow",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    identity = dict(
        schema="ggf-integration-check-v1",
        device=device,
        numerical_versions=versions,
        implementation_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256=sha256(
            json.dumps(
                {
                    p.relative_to(code).as_posix(): sha256(p.read_bytes()).hexdigest()
                    for p in sorted(code.rglob("*.py"))
                },
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        specification_sha256=sha256(
            (code / "bootstrap/SPECIFICATION.json").read_bytes()
        ).hexdigest(),
        workload="Small software-validation fixtures; separate from research endpoints",
    )
    marker = root / "QUICK_CONFIG.json"
    if marker.exists() and json.loads(marker.read_bytes()) != identity:
        raise ValueError("Quick-check source or device changed; choose a new directory")
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise ValueError("Use an empty directory or resume its existing quick check")
    root.mkdir(parents=True, exist_ok=True)
    save(marker, identity)
    sys.path[:0] = [
        str(code / "training"),
        str(code / "training/core"),
        str(code / "sources/controlled_variation"),
    ]
    return root, code


def fixture(split, n=8):
    import numpy as np

    rng = np.random.default_rng(2100 + split)
    x = rng.normal(0, 0.2, (n, 196, 24)).astype("float32")
    masks = []
    for i in range(n):
        fraction = np.zeros((196, 2), np.uint8)
        instances = []
        for c, start in [(0, 0), (1, 72)]:
            if (i + c) % 3 == 0:
                fraction[start : start + 48, c] = 255
                x[i, start : start + 48, c] += 2
                instances.append(
                    dict(
                        category=c,
                        annotation_id=split * 1000 + i * 2 + c,
                        fraction=fraction[:, c].astype("float32") / 255,
                    )
                )
        masks.append(
            dict(fraction=fraction, crowd=np.zeros(196, np.uint8), instances=instances)
        )
    return dict(
        x=x,
        ids=list(range(split * 1000, split * 1000 + n)),
        masks=masks,
        identity="quick-fixture-" + str(split),
    )


def context(root, code, device):
    import torch
    from cd_common import make_store
    from coco_ggf.width_balance import settings as balance_settings
    from scale_plan import settings

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime")
    torch.use_deterministic_algorithms(True)
    rule = settings()
    rule.update(
        width=12,
        images_per_update=2,
        warm_updates=2,
        final_guided_updates=2,
        evaluate_every=1,
        local_save_every=1,
        drive_save_every=1,
        snr=0.0,
        sampling="epochs",
    )
    balance = balance_settings()["balance"]
    balance.update(every=1, reference_batches=1)
    spec = json.loads((code / "bootstrap/SPECIFICATION.json").read_bytes())
    cfg = dict(
        spec["fit_config"],
        device=device,
        category_indices=[0, 1],
        batch=64,
        audit_image_batch=1,
        target_min_images=1,
        decision_min_images=1,
        bootstrap=8,
    )
    store = make_store(
        root / "models",
        root / "scratch",
        dict(scope="integration", rule=rule, device=device),
        drive=False,
    )
    return dict(
        store=store,
        device=device,
        cfg=cfg,
        rule=rule,
        balance_recipe=balance,
        packs=dict(train=fixture(1), select=fixture(2)),
        local=root / "scratch",
        provenance=dict(categories=[dict(name="fixture 0"), dict(name="fixture 1")]),
    )


def public(root, code, device):
    if device != "cuda":
        raise RuntimeError("The public DINOv3 check requires CUDA")
    import clean_room_check

    folder = root / (
        "public_model_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    )
    clean_room_check.run(folder, code)
    return dict(receipt=str(folder / "SMOKE_COMPLETE.json"))


def voc_inputs(root, code, device):
    import numpy as np
    import torch
    from PIL import Image
    from coco_baselines import data
    from coco_stratified import dataset
    import importlib.util

    profile = code / "sources/voc_slics_compute"
    spec = importlib.util.spec_from_file_location(
        "quick_voc_training", profile / "voc_training.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import voc

    ctx = context(root, code, device)
    ctx["options"] = dict(source_output=str(root / "voc_preview/parent"))
    assets, splits = module.assets(ctx)
    records = [
        dict(image=i, official_split=s) for s, ids in splits.items() for i in ids
    ]
    allocation = module.partition(records)
    chosen = (
        [i for i in splits["train"] if allocation[i] == "train"][:2]
        + [i for i in splits["train"] if allocation[i] == "select"][:1]
        + splits["val"][:1]
    )
    config = json.loads((code / "bootstrap/SPECIFICATION.json").read_bytes())
    category_map = voc.mapping(config["categories"])
    crops, masks = [], []
    for i, name in enumerate(chosen):
        with Image.open(assets / "JPEGImages" / (name + ".jpg")) as im:
            geometry = data.geometry(*im.size)
            crops.append(data.transform_image(im.convert("RGB"), geometry))
        with Image.open(assets / "SegmentationClass" / (name + ".png")) as im:
            classes = np.array(im)
        with Image.open(assets / "SegmentationObject" / (name + ".png")) as im:
            objects = np.array(im)
        masks.append(
            voc.mask_pair(classes, objects, geometry, category_map, 2100000000 + i)
        )
    public_receipts = sorted(root.glob("public_model_*/SMOKE_COMPLETE.json"))
    if public_receipts:
        cache = public_receipts[-1].parent / "public_model_cache"
        os.environ["HF_HOME"] = str(cache)
        os.environ["HF_HUB_CACHE"] = str(cache / "hub")
        from huggingface_hub import constants

        constants.HF_HOME = str(cache)
        constants.HF_HUB_CACHE = str(cache / "hub")
        constants.HUGGINGFACE_HUB_CACHE = str(cache / "hub")
    processor, backbone = dataset.backbone(
        config, device, lambda: os.environ.get("HF_TOKEN")
    )
    inputs = processor(
        images=crops, do_resize=False, do_center_crop=False, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        raw = backbone(**inputs).last_hidden_state[:, 5:].float().cpu().half().numpy()
    normalizer = module.streaming_norm(raw, [0, 1])
    normalized = data.normalize(raw.astype("float32"), normalizer)
    if normalized.shape != (4, 196, 768) or not np.isfinite(normalized).all():
        raise ValueError("VOC image, mask or normalization integration failed")
    return dict(
        official_train=len(splits["train"]),
        official_validation=len(splits["val"]),
        images=chosen,
        feature_shape=list(normalized.shape),
        fit_only_normalization_images=2,
        mask_shapes=[list(m["fraction"].shape) for m in masks],
    )


def fits(root, code, device):
    import numpy as np
    import torch
    from coco_baselines import training
    from scale_models import model
    from scale_fit import fit
    from scale_metrics import evaluate, train_targets
    from scale_run import initial
    from cd_common import make_store
    from coco_baselines import data
    from coco_baselines.storage import identity
    from feature_arrays import normalization, normalize_to_file

    raw = fixture(1)["x"]
    expected_norm = data.normalization(raw)
    actual_norm = normalization(raw)
    if identity(actual_norm) != identity(expected_norm):
        raise ValueError("Disk-backed normalization changed training statistics")
    feature_path = root / "quick_normalization.npy"
    normalize_to_file(raw, actual_norm, feature_path)
    if not np.array_equal(np.load(feature_path), data.normalize(raw, expected_norm)):
        raise ValueError("Disk-backed normalization changed feature values")

    ctx = context(root, code, device)
    reports = []
    for kind in ("released_vanilla", "supervised_vanilla", "agf_g"):
        masked = kind == "agf_g"
        w = initial(ctx, "released_vanilla", 75, 32, 21, masked)
        warm = fit(
            ctx, kind + "_warm", "released_vanilla", 75, 32, 21, w, 2, 1, stream=0
        )
        warm = fit(
            ctx, kind + "_warm", "released_vanilla", 75, 32, 21, w, 2, 2, stream=0
        )
        whole = fit(
            ctx,
            kind + "_uninterrupted",
            "released_vanilla",
            75,
            32,
            21,
            w,
            2,
            2,
            stream=0,
        )
        if training.weight_sha(warm["weights"]) != training.weight_sha(
            whole["weights"]
        ):
            raise ValueError("Interrupted/resumed optimizer and RNG state differ")
        m = model("released_vanilla", 75, 12, 32, 24).to(device)
        m.load_state_dict(warm["weights"])
        targets = train_targets(ctx, m, "mapping_" + kind)
        state = fit(
            ctx,
            kind + "_continued",
            "released_vanilla",
            75,
            32,
            21,
            warm["weights"],
            2,
            2,
            targets=None if kind == "released_vanilla" else targets,
            cap=0.6,
            fixed=kind == "supervised_vanilla",
            stream=1,
            readout_targets=targets,
        )
        m.load_state_dict(state["weights"])
        report = evaluate(m, fixture(3), targets, ctx["cfg"], device)
        if kind == "agf_g" and state["balance"]["adaptations"] < 1:
            raise ValueError("Quick check did not exercise the adaptive controller")
        reports.append(
            dict(
                kind=kind,
                updates=state["updates"],
                capture=report["macro_capture"],
                patch_AP=report["macro_ap"],
                VE=report["reconstruction"]["ve"],
                resume_verified=True,
            )
        )
        if kind == "agf_g":
            torch.save(
                dict(weights=state["weights"], targets=targets), root / "quick_ggf.pt"
            )
        del m, state, warm, whole
    split = dict(ctx, cfg=dict(ctx["cfg"], category_indices=[0]))
    split["store"] = make_store(
        root / "split_models",
        root / "split_scratch",
        dict(scope="integration", guide=[0], excluded=[1], device=device),
        drive=False,
    )
    w = initial(split, "released_vanilla", 75, 32, 21, True)
    m = model("released_vanilla", 75, 12, 32, 24).to(device)
    m.load_state_dict(w)
    guide = train_targets(split, m, "guided_mapping")
    state = fit(
        split,
        "guided_only",
        "released_vanilla",
        75,
        32,
        21,
        w,
        2,
        2,
        targets=guide,
        cap=0.6,
    )
    if tuple(state["head"].shape) != (1, 2):
        raise ValueError(
            "Excluded-category experiment acquired an extra supervised head"
        )
    m.load_state_dict(state["weights"])
    excluded = dict(split, cfg=dict(split["cfg"], category_indices=[1]))
    readouts = train_targets(excluded, m, "post_fit_excluded_readout")
    evaluate(m, fixture(3), readouts, excluded["cfg"], device)
    del m, state
    probes = []
    for method in (
        "released_vanilla",
        "released_grassmannian",
        "released_group_lasso",
        "paper_vanilla",
        "paper_grassmannian",
        "paper_group_lasso",
    ):
        torch.manual_seed(21)
        m = model(method, 75, 12, 32, 24).to(device)
        value, _ = m.loss(torch.tensor(fixture(1)["x"][0], device=device))
        if not torch.isfinite(value):
            raise ValueError("Nonfinite baseline objective: " + method)
        value.backward()
        probes.append(method)
        del m
    for groups, k in [(19, 8), (1024, 16), (4096, 32)]:
        m = model("released_vanilla", groups, 12, k, 24).to(device)
        value, _ = m.loss(torch.tensor(fixture(1)["x"][0, :8], device=device))
        value.backward()
        del m
    save(
        root / "QUICK_METRICS.json", dict(scope="integration fixtures", models=reports)
    )
    return dict(
        models=reports,
        baseline_objectives=probes,
        group_counts=[19, 75, 1024, 4096],
        disk_backed_normalization_verified=True,
        category_split=dict(
            guide=[0],
            excluded=[1],
            supervised_heads=1,
            excluded_readout_fitted_after_training=True,
        ),
    )


def native(root, code, device):
    import torch
    from cd_train import oriented_initializer
    from coco_ggf.slics_closeout import support, atom_sweep
    from coco_ggf.slics_repair import Solver
    from training_precision import JointPrecisionRetry, precision_atom_sweep

    ctx = context(root, code, device)
    D = oriented_initializer(ctx).to(device)
    x = torch.tensor(ctx["packs"]["train"]["x"][:2], device=device).reshape(-1, 24)
    allowed, known = support(
        ctx["packs"]["train"]["masks"][:2], [0, 1], dict(rank=12), "mask", device
    )
    x, allowed = x[known], allowed[known]
    cfg = dict(
        nnls_iterations=2000, nnls_tolerance=1e-5, nnls_fallback_iterations=20000
    )
    retry = JointPrecisionRetry(Solver.solve)
    coefficients, info = retry(Solver(D, cfg), x, allowed)
    changed, _ = precision_atom_sweep(atom_sweep, x, D, coefficients, allowed, False)
    coefficients, final = retry(Solver(changed, cfg), x, allowed)
    if final["max_relative_kkt"] > 1e-5 or not torch.isfinite(changed).all():
        raise ValueError("Native training or strict NNLS certificate failed")
    torch.save(dict(D=changed.cpu(), solver=cfg), root / "quick_slics.pt")
    return dict(
        atom_update=True,
        strict_KKT=final["max_relative_kkt"],
        coefficients=int(coefficients.numel()),
    )


def concept(root, code, device):
    import numpy as np
    import torch
    import concept_sae as native

    torch.manual_seed(21)
    m = native.build(
        native.source(code / "training/concept_sae_native.py"),
        2,
        p=196,
        d=24,
        seg=14,
        free=4,
    ).to(device)
    losses = []
    pack = fixture(1)
    for stage in (1, 2, 3):
        part = native.configure(m, stage)
        optimizer = torch.optim.Adam(
            [p for p in part.parameters() if p.requires_grad], lr=0.001
        )
        value = native.optimizer_step(
            m, part, optimizer, pack, [0, 1], [0, 1], device, stage, microbatch=1
        )
        if not np.isfinite(value):
            raise ValueError("Nonfinite Concept-SAE stage")
        losses.append(value)
    torch.save(m.state_dict(), root / "quick_concept.pt")
    return dict(stages=[1, 2, 3], losses=losses, native_source_sha256=native.SHA)


def analysis(root, code, device):
    import importlib.util
    import numpy as np
    from analysis import rank_stats
    from coco_ggf.wider_groups import spectrum
    from cv_math import render, pca_from_stats, project, fit_ridge, predict

    path = code / "sources/geometry_consistency/study_math.py"
    spec = importlib.util.spec_from_file_location("quick_geometry", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rng = np.random.default_rng(21)
    x, y = rng.normal(size=(64, 24)), rng.normal(size=(32, 24))
    geometry = mod.geometry(x, y, np.arange(64) % 2 == 0, np.arange(32) % 2 == 0)
    rank = rank_stats(x[:, :12].T @ x[:, :12])
    headroom = spectrum(x, y, [3, 4, 8, 16])
    rgb = np.full((224, 224, 3), 140, np.uint8)
    mask = np.zeros((224, 224), np.uint8)
    mask[60:160, 60:160] = 1
    shapes = []
    for attribute, level in [
        ("rotation", 15.0),
        ("scale", 0.875),
        ("brightness", 0.8),
        ("background", 1.2),
    ]:
        image, fraction = render(rgb, mask, attribute, level)
        image.save(root / (attribute + ".png"))
        shapes.append(list(fraction.shape))
    z, D = x[:, :12], rng.normal(size=(12, 24))
    pca = pca_from_stats(len(z), z.sum(0), z.T @ z, D)
    coords = project(z, D, pca)
    probe = fit_ridge(coords, x[:, 0])
    if not np.isfinite(predict(probe, coords)).all():
        raise ValueError("Nonfinite frozen probe")
    return dict(
        geometry=geometry,
        entropy_rank=rank["entropy_rank"],
        headroom=headroom,
        transformations=shapes,
        PCA=list(coords.shape),
    )


def presence(root, code, device):
    import numpy as np
    import torch
    import original_concentration as metrics
    from scale_models import model

    saved = torch.load(root / "quick_ggf.pt", map_location="cpu", weights_only=False)
    m = model("released_vanilla", 75, 12, 32, 24).to(device)
    m.load_state_dict(saved["weights"])
    pack = fixture(4, 40)
    with torch.no_grad():
        response = np.stack(
            [
                m.encode(torch.tensor(x, device=device)).norm(dim=-1).cpu().numpy()
                for x in pack["x"]
            ]
        )
    scores = metrics.mean_top4(response)
    labels = {iid: "yes" if i % 3 == 0 else "no" for i, iid in enumerate(pack["ids"])}
    selection = metrics.calibrate_description(
        scores, pack["ids"], pack["ids"][:20], labels
    )
    report = metrics.evaluate_category_synthetic_or_after_gates(
        selection, scores, pack["ids"], pack["ids"][20:], labels
    )
    save(root / "QUICK_PRESENCE.json", report)
    return dict(
        calibration=20,
        audit=20,
        labels="Known synthetic fixture labels",
        quantiles=len(selection["quantiles"]),
    )


def synthetic(root, code, device):
    from fresh_zoo import modules, store

    focused, native, data = modules(code)
    cfg = dict(
        dimension=16,
        factors=8,
        groups=16,
        block_rank=4,
        active_factors=4,
        train_rows=64,
        select_rows=64,
        calibration_rows=128,
        test_rows=128,
        normalization_rows=128,
        batch=16,
        epochs=2,
        warm_epochs=1,
        warmup=0.05,
        log_epochs=1,
        snr=0.0,
        save_updates=2,
    )
    ds = data.generate(cfg, "goodfire", "independent", seed=701)
    outputs = []
    for method in ("vanilla", "grassmannian", "group_lasso"):
        journal = store(root / "zoo", native, cfg, dict(scope="integration"), method)
        state = native.train_native(
            journal,
            "fit",
            method,
            ds["packs"]["train"]["x"],
            ds["packs"]["select"]["x"],
            0,
            0.0004,
            device,
        )
        report = native.audit(journal, "fit", state["weights"], method, ds, device)
        outputs.append(dict(method=method, updates=state["updates"]))
    journal = store(root / "zoo", native, cfg, dict(scope="integration"), "ggf")
    state = focused.fit(journal, "fit", cfg, ds, 0, 0.0004, device=device)
    outputs.append(dict(method="ggf", updates=state["updates"]))
    return dict(models=outputs)


def compute(root, code, device):
    import numpy as np
    from coco_baselines import training
    from scale_models import model

    sys.path.append(str(code / "sources/voc_slics_compute"))
    from compute_probe import benchmark, bsf_factory

    ctx = context(root, code, device)
    pack = fixture(1)
    features = np.zeros((8, 196, 768), np.float32)
    features[:, :, :24] = pack["x"]
    ctx["packs"]["train"] = dict(pack, x=features)
    m = model("released_vanilla", 19, 12, 8, 768)
    state = dict(weights=training.weights(m), head=None, optimizer=None)
    row = dict(groups=19, active_groups=8, lr=0.001, cap=0.6)
    del m
    output = benchmark(
        bsf_factory(ctx, row, state, [0], True, False), 1, repeats=2, flop_repeats=1
    )
    if (
        len(output["seconds"]) != 2
        or output["flop_probes"][0]["registered_operator_FLOPs"] <= 0
    ):
        raise ValueError("Execution profiling or FLOP accounting failed")
    save(root / "QUICK_COMPUTE.json", output)
    return dict(
        repeats=2,
        images=1,
        registered_FLOPs=output["flop_probes"][0]["registered_operator_FLOPs"],
        full_FLOPs_available=False,
    )


def historical(root, code, device):
    """Small updates through the fit kernels used by the older studies."""
    import copy
    import torch
    from historical_inputs import activate

    activate(code, "components")
    from coco_baselines import training
    from coco_adaptation import core
    from coco_ggf import (
        large_mask_fit,
        mask_recovery_fit,
        width_balance,
        width_balance_fit,
        warm_capacity,
    )
    from coco_ggf import next_fit, capacity_fit

    ctx = context(root, code, device)
    ctx["base"] = dict(
        groups=4,
        block_rank=3,
        active_groups=2,
        epochs=1,
        batch=64,
        snr=0.0,
        warmup=0.1,
        save_updates=1,
        local_save_updates=1,
        sparsity_tolerance=0.2,
        lrs=[0.003],
    )
    ctx["cfg"].update(
        updates=2, evaluate_every=1, diagnostic_grad_every=1, local_save_every=1
    )
    ctx["recipe"] = width_balance.settings()
    ctx["recipe"]["balance"].update(every=1, reference_batches=1)
    x = ctx["packs"]["train"]["x"].reshape(-1, 24)
    val = ctx["packs"]["select"]["x"].reshape(-1, 24)
    warm = training.fit(
        ctx["store"],
        "legacy_native",
        "vanilla",
        x,
        val,
        ctx["base"],
        21,
        0.003,
        device,
        max_updates=2,
    )
    resumed = training.fit(
        ctx["store"],
        "legacy_native",
        "vanilla",
        x,
        val,
        ctx["base"],
        21,
        0.003,
        device,
        max_updates=3,
    )
    if resumed["updates"] != 3:
        raise ValueError("Historical warm-up resume failed")
    model = core.new_model("vanilla", ctx["base"], warm["best"], device)
    from coco_baselines import metrics

    pack = ctx["packs"]["train"]
    arrays = metrics.audit_arrays(
        model, pack["x"], pack["masks"], pack["ids"], ctx["cfg"], device
    )
    targets = core.choose_targets(arrays, ctx["cfg"], ctx["provenance"]["categories"])
    reference = core.validation(
        model, ctx["packs"]["select"], targets, ctx["cfg"], device
    )["reconstruction"]["mse"]
    guided = large_mask_fit.fit_guided(
        ctx,
        "legacy_guided",
        warm["best"],
        targets,
        True,
        71021,
        reference_mse=reference,
    )
    mask_recovery_fit.fit_recovery(
        ctx,
        "legacy_recovery",
        guided["weights"],
        targets,
        True,
        82021,
        reference_mse=reference,
        initial_head=guided["head"],
    )
    balanced = width_balance_fit.fit(
        ctx,
        "legacy_balanced",
        warm["best"],
        targets,
        2,
        71021,
        True,
        True,
        reference_mse=reference,
    )
    next_fit.fit(
        ctx,
        "legacy_projected",
        balanced["weights"],
        targets,
        2,
        93021,
        True,
        True,
        reference,
        initial_head=balanced["head"],
        initial_balance=copy.deepcopy(balanced["balance"]),
        initial_optimizer=balanced["optimizer"],
        pcgrad=True,
        decoder_pcgrad=True,
    )
    capacity_fit.fit(
        ctx,
        "legacy_capacity",
        balanced["weights"],
        targets,
        2,
        97021,
        True,
        True,
        reference,
        initial_head=balanced["head"],
        initial_balance=copy.deepcopy(balanced["balance"]),
        initial_optimizer=balanced["optimizer"],
    )
    torch.manual_seed(21)
    base5 = dict(ctx["base"], block_rank=5)
    model5 = training.make_model("vanilla", base5, 24).to(device)
    optimizer = torch.optim.Adam(model5.parameters(), lr=0.003)
    loss, _ = model5.loss(torch.tensor(x[:8], device=device))
    loss.backward()
    optimizer.step()
    weights = training.weights(model5)
    moments = copy.deepcopy(optimizer.state_dict())
    grown = warm_capacity.widen_weights(weights, 4, 8, 21)
    grown_opt = warm_capacity.widen_optimizer(moments, weights, 4, 8, False)
    model8 = core.new_model("vanilla", dict(base5, block_rank=8), grown, device)
    with torch.no_grad():
        before, _ = model5(torch.tensor(x[:8], device=device))
        after, _ = model8(torch.tensor(x[:8], device=device))
    if not torch.allclose(before, after, rtol=1e-5, atol=1e-6):
        raise ValueError("Width growth changed the initial reconstruction")
    if len(grown_opt["state"]) != len(moments["state"]):
        raise ValueError("Width growth dropped Adam state")
    return dict(
        warm_resume=True,
        fit_kernels=[
            "native",
            "guided",
            "recovery",
            "bounded_weights",
            "PCGrad",
            "capacity",
        ],
        width_growth=dict(
            old=5,
            new=8,
            initial_reconstruction_preserved=True,
            optimizer_states=len(grown_opt["state"]),
        ),
    )


def cards(root, code, device):
    from PIL import Image
    from coco_baselines import data
    from prepare_evaluation_reviews import card

    spec = json.loads((code / "bootstrap/SPECIFICATION.json").read_bytes())
    images = root / "card_images"
    images.mkdir(exist_ok=True)
    records = {}
    views = []
    for iid in spec["splits"]["train"][:6]:
        source = data.download(
            data.COCO_BUCKET + f"train2017/{iid:012d}.jpg", images / f"{iid}.jpg"
        )
        with Image.open(source) as im:
            crop = data.transform_image(im.convert("RGB"), data.geometry(*im.size))
        png = images / f"{iid}.png"
        crop.save(png)
        records[iid] = dict(
            status="ready",
            crop_path=png.relative_to(root).as_posix(),
            crop_sha256=sha256(png.read_bytes()).hexdigest(),
        )
        views.append(dict(image_id=iid, patches=[0, 15, 30, 45]))
    path = root / "quick_card.png"
    invalid = card(root, path, "Integration layout check", views, records)
    if any(invalid):
        raise ValueError("A generated card view is invalid")
    with Image.open(path) as image:
        if image.size != (1344, 364):
            raise ValueError("Card dimensions differ")
    return dict(
        views=6, card=str(path), judgments="Not collected in the integration check"
    )


def figures(root, code, device):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = json.loads((root / "QUICK_METRICS.json").read_bytes())
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        [r["kind"] for r in metrics["models"]],
        [r["capture"] for r in metrics["models"]],
    )
    ax.set(ylabel="Fixture Capture", title="Pipeline integration check")
    ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(root / "quick_metrics.pdf")
    fig.savefig(root / "quick_metrics.png")
    plt.close(fig)
    for name in ("geometry", "architecture", "interpretability", "voc"):
        run_command(
            [
                sys.executable,
                str(code / "workflows.py"),
                "run",
                "figures." + name,
                "--execute",
            ],
            root / "logs",
            "figures_" + name,
        )
    return dict(
        fixture_plot=str(root / "quick_metrics.pdf"), saved_paper_figure_families=4
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--phase", choices=["all", *STAGES], default="all")
    a = p.parse_args()
    root, code = setup(a.root, a.code_root, a.device)
    if a.phase == "all":
        for stage in STAGES:
            receipt = root / "checks" / (stage + ".json")
            if receipt.exists() and json.loads(receipt.read_bytes()).get("complete"):
                print("Reusing completed quick section:", stage, flush=True)
                continue
            run_command(
                [
                    sys.executable,
                    str(Path(__file__)),
                    "--root",
                    str(root),
                    "--code-root",
                    str(code),
                    "--device",
                    a.device,
                    "--phase",
                    stage,
                ],
                root / "logs",
                stage,
            )
        receipts = {
            name: json.loads((root / "checks" / (name + ".json")).read_bytes())
            for name in STAGES
        }
        save(
            root / "QUICK_COMPLETE.json",
            dict(
                complete=True,
                scope="Software integration check",
                research_run=False,
                sections=receipts,
                time=datetime.now(timezone.utc).isoformat(),
            ),
        )
        print(
            "All quick integration sections passed:",
            root / "QUICK_COMPLETE.json",
            flush=True,
        )
    else:
        import torch

        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        result = globals()[a.phase](root, code, a.device)
        save(
            root / "checks" / (a.phase + ".json"),
            dict(complete=True, stage=a.phase, details=result),
        )
        print("Quick section passed:", a.phase, flush=True)


if __name__ == "__main__":
    main()
