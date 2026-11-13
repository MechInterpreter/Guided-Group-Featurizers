"""Additional fixed 75-group training orders; historical releases stay read-only."""

import argparse, copy, gc, hashlib, json, os, signal, threading
from pathlib import Path
import numpy as np
import torch
import all75_run, cd_common, cd_data, cd_train
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from coco_ggf import matched_baselines as matched

HERE = Path(__file__).resolve().parent
SEEDS = [24, 25, 26, 27]
read = lambda p: json.loads(Path(p).read_text(encoding="utf8"))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def configure(regime):
    all75_run.configure(regime)
    cd_common.SPEC.update(
        seeds=SEEDS.copy(), revision="all75-" + regime + "-seeds24-27-v1"
    )
    assert (
        cd_common.SPEC["groups"],
        cd_common.SPEC["width"],
        cd_common.SPEC["top_k"],
    ) == (75, 12, 32)


def expected(regime):
    return (
        {(s, 0, k) for s in SEEDS for k in ["vanilla", "supervised_vanilla"]}
        if regime == "full_category"
        else {(s, 0, "vanilla") for s in SEEDS}
        | {
            (s, f, k)
            for s in SEEDS
            for f in [0, 1]
            for k in ["ggf", "supervised_vanilla"]
        }
    )


def frozen_selection(ctx, fold, kind):
    """Read seed21's actual locked weights, without creating a seed21 store."""
    assert kind == "supervised_vanilla"
    pins = read(HERE / "ADDITIONAL_SEEDS_PINS.json")
    r = next(
        r
        for r in pins["original_banks"]
        if r["regime"] == ctx["options"]["regime"]
        and r["seed"] == 21
        and r["kind"] == kind
        and r["fold"] == fold
    )
    output = all75_run.mounted_path(ctx["options"]["drive_root"], r["output"])
    store = memory.FrozenStore(
        output / "checkpoints" / r["run_id"][:20],
        Path(ctx["options"]["local"]) / "read_selected_weights",
    )
    state = memory.required(store, "frozen")
    if training.weight_sha(state["weights"]) != r["weights_sha"]:
        raise ValueError("Original seed21 selection source changed")
    selection = memory.required(store, "fixed_weight_selection")
    if selection["weights"] not in [
        [1.0, 0.05, 0.01],
        [1.0, 0.2, 0.2],
        [1.0, 0.4, 0.4],
    ]:
        raise ValueError("Unknown frozen supervised weight pair")
    ctx["store"].write(
        f"fixed_weight_source_fold{fold}.json",
        dict(
            original_run_id=store.run_id,
            original_weights_sha=r["weights_sha"],
            selection=selection,
            reused_not_retuned=True,
        ),
    )
    return selection


class InitializerReuse:
    """Only the exact common initializer may fall through to the old store."""

    def __init__(self, job, old):
        self.job, self.old = job, old

    def __getattr__(self, name):
        return getattr(self.job, name)

    def load(self, key):
        value = self.job.load(key)
        if value is None and key == "native_svd12":
            value = memory.required(self.old, key)
            if training.weight_sha({"D": value["D"]}) != value["sha"]:
                raise ValueError("Common native initializer changed")
        return value


def legacy_context(ctx):
    """Recreate original full-category GGF/SLiCS context, not split training."""
    import supervised_study as study
    from coco_ggf.width_balance import settings

    pins = read(HERE / "ADDITIONAL_SEEDS_PINS.json")
    drive = Path(ctx["options"]["drive_root"])
    parent = memory.FrozenStore(
        drive
        / "GGF_COCO_supervised_comparison_v1/checkpoints"
        / pins["legacy_manifest"]["run_id"][:20],
        Path(ctx["options"]["local"]) / "read_full_parent",
    )
    if dict(run_id=parent.run_id, config=parent.config) != pins["legacy_manifest"]:
        raise ValueError("Exact original full-category parent manifest required")
    if identity(ctx["norm"]) != parent.config["normalization_sha"] or any(
        parent.config[k] != ctx["packs"][v]["identity"]
        for k, v in [("train_identity", "train"), ("development_identity", "select")]
    ):
        raise ValueError("Full-category parent data/normalization differs")
    # The full-category recipe originally came from this frozen scale source.
    source_run = study.STUDY_RUN
    source = memory.FrozenStore(
        drive / "AGF_COCO_scale_study_v1/checkpoints" / source_run[:20],
        Path(ctx["options"]["local"]) / "read_scale_recipe",
    )
    if source.run_id != source_run or source.config["source_sha"] != study.CORE_SHA:
        raise ValueError("Original GGF full-category recipe source differs")
    rule = copy.deepcopy(source.config["recipe"])
    if (rule["width"], rule["images_per_update"], rule["warm_updates"]) != (
        12,
        8,
        7640,
    ):
        raise ValueError("Unexpected original full-category recipe")
    result = dict(
        ctx,
        core_rule=rule,
        rule=rule,
        local=Path(ctx["options"]["local"]) / "legacy_full",
        active_key=None,
        balance_recipe=copy.deepcopy(settings()["balance"]),
        cfg=dict(
            ctx["cfg"],
            batch=512,
            audit_image_batch=1,
            bootstrap=1000,
            max_audit_groups=4,
            preview_images=0,
        ),
    )
    result["local"].mkdir(parents=True, exist_ok=True)
    result["store"] = InitializerReuse(
        cd_common.make_store(
            Path(ctx["options"]["output"]) / "legacy_full_training",
            result["local"] / "journal",
            dict(
                extension=sha(HERE / "ADDITIONAL_SEEDS_PLAN.json"),
                parent=parent.run_id,
                source_recipe=source.config,
                source_code=ctx["options"]["experiment_sha"],
            ),
            ctx["store"].require_drive,
        ),
        parent,
    )
    return result, parent


def install_legacy_precision(store):
    """Apply exactly the previously used strict-KKT precision recovery."""
    import supervised_study as study, object_schedule
    from training_precision import JointPrecisionRetry, patch_paths
    from precision_retry import PrecisionRetry
    from coco_ggf.slics_repair import Solver

    if getattr(Solver.solve, "_additional_precision", False):
        raise RuntimeError("Precision overlay installed twice")
    partial, joint = (
        PrecisionRetry(study.partial_nnls),
        JointPrecisionRetry(Solver.solve),
    )
    partial.store = joint.store = store
    patch_paths(study, object_schedule)

    def solve(self, x, allowed=None, force_fallback=False):
        return joint(self, x, allowed, force_fallback=force_fallback)

    solve._additional_precision = True
    Solver.solve = solve
    study.partial_nnls = partial


def verify_legacy_recipe(ctx, parent):
    """Recompute existing warm/final resume fingerprints, with writes forbidden."""
    import supervised_study as study

    prior = dict(ctx, store=parent)
    warm, targets = study.small_warm(prior, 24)
    state = study.fit(
        study.small_context(prior),
        "small75_s24_lr10000_ggf",
        "released_vanilla",
        75,
        32,
        24,
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
    pin = next(
        r
        for r in read(HERE / "ADDITIONAL_SEEDS_PINS.json")["five_seed_models"]
        if r["seed"] == 24 and r["key"] == "small75_s24_lr10000_ggf"
    )
    if training.weight_sha(state["weights"]) != pin["weights_sha"]:
        raise ValueError("Original full GGF recipe replay differs")
    ctx["store"].write(
        "ORIGINAL_FULL_RECIPE_PARITY.json",
        dict(
            complete=True,
            original_seed=24,
            original_resume_fingerprints_verified=True,
            original_weights_sha=pin["weights_sha"],
            parent_writes_forbidden=True,
        ),
    )


def export_legacy(ctx, parent, seed, kind):
    """Freeze actual fixed endpoints; never substitute a selected checkpoint."""
    import supervised_study as study

    pins = read(HERE / "ADDITIONAL_SEEDS_PINS.json")
    key = f"small75_s{seed}_lr10000_ggf" if kind == "ggf" else f"native_slics12_s{seed}"
    if seed in [24, 25]:
        state = memory.required(parent, key)
        pin = next(
            r for r in pins["five_seed_models"] if r["seed"] == seed and r["key"] == key
        )
        weights = {"D": state["D"]} if kind == "slics" else state["weights"]
        if training.weight_sha(weights) != pin["weights_sha"]:
            raise ValueError(
                "Existing seed24/25 endpoint differs from fixed-budget catalog"
            )
        origin = "verified_existing_fixed_endpoint"
    else:
        if kind == "ggf":
            warm, targets = study.small_warm(ctx, seed)
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
            del warm
        else:
            study.native_fit(ctx, seed)
            state = memory.required(ctx["store"], key)
        weights = {"D": state["D"]} if kind == "slics" else state["weights"]
        origin = "new_fixed_endpoint_original_full_recipe"
    if state["updates"] != (10640 if kind == "slics" else 3000):
        raise ValueError("Unfinished or checkpoint-selected legacy endpoint")
    if kind == "ggf":
        all75_run.check_weights(weights)
    elif tuple(weights["D"].shape) != (900, 768):
        raise ValueError("SLiCS must be 75x12")
    folder = Path(ctx["options"]["output"]) / "native_exports"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (key + ".pt")
    digest = training.weight_sha(weights)
    output = (
        dict(D=weights["D"], updates=10640)
        if kind == "slics"
        else dict(weights=weights, updates=3000, groups=75, active=32)
    )
    if path.exists():
        actual = torch.load(path, map_location="cpu", weights_only=False)
        if (
            training.weight_sha(
                {"D": actual["D"]} if kind == "slics" else actual["weights"]
            )
            != digest
        ):
            raise ValueError("Immutable exported native weights changed")
    else:
        temp = path.with_suffix(".pending.pt")
        torch.save(output, temp)
        os.replace(temp, path)
    row = dict(
        seed=seed,
        collector_method="slics" if kind == "slics" else "ggf75",
        path=str(path),
        file_sha256=sha(path),
        weights_sha=digest,
        groups=75,
        width=12,
        normalization_identity=identity(ctx["norm"]),
        source_key=key,
        source_status=origin,
        source_run=parent.run_id if seed in [24, 25] else ctx["store"].run_id,
        total_updates=10640,
        training_protocol="original_full_category_fixed_endpoint",
    )
    del state, output, weights
    gc.collect()
    torch.cuda.empty_cache()
    return row


def verify_training(receipt, regime, source_sha, protocol_sha):
    if (
        receipt.get("complete") is not True
        or receipt.get("regime") != regime
        or receipt.get("source_sha") != source_sha
        or receipt.get("protocol_sha256") != protocol_sha
        or receipt.get("required_seeds") != SEEDS
        or receipt.get("scope") != "additional_seeds24_27"
        or receipt.get("amendment_sha256") != sha(HERE / "ADDITIONAL_SEEDS_PLAN.json")
    ):
        raise ValueError("Additional seed training provenance mismatch")
    rows = receipt["new_endpoints"]
    if len(rows) != len(expected(regime)) or {
        (r["seed"], r["fold"], r["kind"]) for r in rows
    } != expected(regime):
        raise ValueError("Incomplete/duplicate additional seed matrix")
    if any(
        (r["groups"], r["width"], r["top_k"], r["total_updates"]) != (75, 12, 32, 10640)
        for r in rows
    ):
        raise ValueError("Wrong group quantity or training budget")
    native = receipt.get("reused_endpoints", [])
    wanted = (
        {(s, k) for s in SEEDS for k in ["ggf75", "slics"]}
        if regime == "full_category"
        else set()
    )
    if (
        len(native) != len(wanted)
        or {(r["seed"], r["collector_method"]) for r in native} != wanted
    ):
        raise ValueError("Incomplete full-category native matrix")
    if any(
        (r["groups"], r["width"], r["total_updates"]) != (75, 12, 10640) for r in native
    ):
        raise ValueError("Wrong native capacity or endpoint")
    if receipt.get("native_references"):
        raise ValueError("No new unmatched references in this extension")
    return receipt


def run(options, require_drive=True):
    configure(options["regime"])
    cd_train.training_context = all75_run.training_context
    if not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU")
    torch.set_num_threads(1)
    torch.manual_seed(21)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    lease = cd_common.Lease(options["output"])
    lease.acquire()
    stop = threading.Event()
    errors = []
    ctx = None
    legacy = None

    def beat():
        while not stop.wait(30):
            try:
                lease.beat()
            except Exception as exc:
                errors.append(exc)
                os.kill(os.getpid(), signal.SIGTERM)
                return

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()

    def interrupted(*_):
        raise KeyboardInterrupt("Retaining last complete training generation")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        ctx = cd_data.prepare(options, require_drive)
        if options["regime"] == "full_category":
            ctx["folds"] = [dict(fold=0, guide=list(ctx["eligible"]), excluded=[])]
        payload = dict(
            protocol=read(HERE / "PROTOCOL.json"),
            amendment=read(HERE / "ADDITIONAL_SEEDS_PLAN.json"),
            runtime_spec=copy.deepcopy(cd_common.SPEC),
            regime=options["regime"],
            folds=ctx["folds"],
            normalization=identity(ctx["norm"]),
            train=ctx["packs"]["train"]["identity"],
            development=ctx["packs"]["select"]["identity"],
            source_sha=options["experiment_sha"],
        )
        ctx["store"] = cd_common.make_store(
            Path(options["output"]) / "training",
            Path(options["local"]) / "training_journal",
            payload,
            require_drive,
        )
        ctx["store"].write("PROTOCOL.json", payload)
        # Verify all selected weights before spending GPU time.
        for fold in range(len(ctx["folds"])):
            frozen_selection(ctx, fold, "supervised_vanilla")
        native = []
        if options["regime"] == "full_category":
            legacy, parent = legacy_context(ctx)
            verify_legacy_recipe(legacy, parent)
            install_legacy_precision(legacy["store"])
            for seed in SEEDS:
                for kind in ["ggf", "slics"]:
                    if errors:
                        raise errors[0]
                    print("FULL NATIVE", seed, kind, flush=True)
                    native.append(export_legacy(legacy, parent, seed, kind))
                source = parent if seed in [24, 25] else legacy["store"]
                warm = memory.required(source, f"small75_s{seed}_warm")
                final = memory.required(source, f"small75_s{seed}_lr10000_ggf")
                slics = memory.required(source, f"native_slics12_s{seed}")
                if (
                    slics["phase_batch_shas"]["warm"] != warm["sampled_batches_sha"]
                    or slics["sampled_batches_sha"] != final["sampled_batches_sha"]
                ):
                    raise ValueError("Original GGF/SLiCS image schedules differ")
                del warm, final, slics
                gc.collect()
                torch.cuda.empty_cache()
        rows = []
        for seed in SEEDS:
            tasks = [(0, "vanilla")] + [
                (f, k)
                for f in range(len(ctx["folds"]))
                for k in (
                    ["supervised_vanilla"]
                    if options["regime"] == "full_category"
                    else ["supervised_vanilla", "ggf"]
                )
            ]
            for fold, kind in tasks:
                if errors:
                    raise errors[0]
                print(
                    "ADDITIONAL75 FIT",
                    dict(
                        regime=options["regime"],
                        seed=seed,
                        fold=fold,
                        kind=kind,
                        groups=75,
                    ),
                    flush=True,
                )
                row = all75_run.checked_endpoint(
                    ctx, cd_train.fit_bsf(ctx, fold, seed, kind)
                )
                rows.append(row)
                same = [r for r in rows if r["seed"] == seed]
                if any(
                    len({r[k] for r in same}) != 1
                    for k in ["continuation_batches_sha", "continuation_noise_sha"]
                ):
                    raise ValueError("Paired BSF schedules differ")
                ctx["store"].write(
                    "ENDPOINT_PROGRESS.json",
                    dict(models=rows, native=native, complete=False),
                )
                gc.collect()
                torch.cuda.empty_cache()
        receipt = dict(
            complete=True,
            scope="additional_seeds24_27",
            required_seeds=SEEDS,
            regime=options["regime"],
            source_sha=options["experiment_sha"],
            protocol_sha256=sha(HERE / "PROTOCOL.json"),
            amendment_sha256=sha(HERE / "ADDITIONAL_SEEDS_PLAN.json"),
            new_endpoints=rows,
            reused_endpoints=native,
            native_references=[],
            evaluation_status="pending_collection_and_new_blinded_reviews",
        )
        verify_training(
            receipt,
            options["regime"],
            options["experiment_sha"],
            sha(HERE / "PROTOCOL.json"),
        )
        matched.assert_sources(ctx)
        ctx["store"].write("TRAINING_COMPLETE.json", receipt)
        cd_common.write_json(
            Path(options["output"]) / "TRAINING_COMPLETE.json", receipt
        )
        print("ADDITIONAL75 TRAINING COMPLETE", options["regime"], flush=True)
    finally:
        try:
            if ctx is not None:
                for s in {
                    str(s.root): s
                    for s in [ctx["store"]] + ctx.get("writable_stores", [])
                }.values():
                    cd_common.flush(s)
            if legacy is not None:
                cd_common.flush(legacy["store"])
        finally:
            stop.set()
            thread.join(timeout=2)
            lease.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run(read(parser.parse_args().config))
