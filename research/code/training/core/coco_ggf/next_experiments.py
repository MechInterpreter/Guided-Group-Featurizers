"""Prioritized, independently frozen three-seed experiments on cached COCO."""

import copy, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from coco_baselines import training
from coco_baselines.storage import identity
from coco_adaptation import core, inputs
from coco_coordinated import run as memory
from . import pcgrad_continuation as previous, next_fit as engine
from .next_warm import fit_warm

base = previous.base
matched = base.matched
SEEDS = [21, 22, 23]
PRIORITY = ["width5", "pcgrad_both", "longer"]
PREVIOUS_RUN = "32dcb7bd6e9ccc3ad24fa9744a9e390229af6b2eea773fff813a4efdeabe0f41"
STAGE_ARMS = {
    "width5": ["mask5", "vanilla5"],
    "pcgrad_both": ["pcgrad_both"],
    "longer": ["longer_guided", "longer_vanilla"],
}
STAGE_CONTROLS = {
    "width5": ["cap040", "vanilla4"],
    "pcgrad_both": ["ggf_continue", "ggf_pcgrad", "vanilla_continue", "cap040"],
    "longer": ["ggf_continue", "vanilla_continue"],
}
PAIRS = {
    "width5": [("mask5", "vanilla5"), ("mask5", "cap040"), ("vanilla5", "vanilla4")],
    "pcgrad_both": [
        ("pcgrad_both", "ggf_continue"),
        ("pcgrad_both", "ggf_pcgrad"),
        ("pcgrad_both", "vanilla_continue"),
        ("pcgrad_both", "cap040"),
    ],
    "longer": [
        ("longer_guided", "longer_vanilla"),
        ("longer_guided", "ggf_continue"),
        ("longer_vanilla", "vanilla_continue"),
    ],
}
LABELS = {
    "mask5": "AGF-G 5D / cap 0.40",
    "vanilla5": "Vanilla BSF 5D",
    "pcgrad_both": "AGF-G 4D / encoder + decoder PCGrad",
    "longer_guided": "AGF-G 4D / longer ordinary continuation",
    "longer_vanilla": "Vanilla BSF 4D / longer continuation",
    "cap040": "AGF-G 4D / saved cap 0.40",
    "vanilla4": "Vanilla BSF 4D / saved control",
    "ggf_continue": "AGF-G 4D / saved ordinary continuation",
    "ggf_pcgrad": "AGF-G 4D / saved encoder PCGrad",
    "vanilla_continue": "Vanilla BSF 4D / saved continuation",
}


def settings():
    return dict(
        version=1,
        study="agf_prioritized_three_experiments",
        seeds=SEEDS,
        priority=PRIORITY,
        groups=1024,
        active_groups=16,
        balance=copy.deepcopy(base.bound.settings()["balance"]),
        width5=dict(
            warm_epochs=40,
            warm_updates=7640,
            phases=[2500, 500],
            phase_seeds=[71000, 82000],
            optimizer="Reset model Adam at phase boundary; retain heads/controller, matching saved 4D cap 0.40.",
            initialization="Preserve saved four TRAIN axes exactly; append the orthogonal fifth TRAIN eigen-direction.",
        ),
        pcgrad_both=dict(
            updates=3000,
            seed_offset=93000,
            start="Saved final cap 0.40",
            projection="Encoder R/C/D and decoder R/C projected separately; classifier stays ordinary; independent projection RNGs.",
            controls="Reuse completed ordinary, encoder-only PCGrad and continued Vanilla; no control retraining.",
        ),
        longer=dict(
            updates=3000,
            seed_offset=94000,
            start="Saved final ordinary continuation and matched Vanilla continuation",
        ),
        endpoint="Final endpoints only. No test-selected model. Retain failed original native-MSE x 1.02 limits.",
        evaluation="Freeze all seeds of each experiment, evaluate, export, then proceed to next experiment. Unchanged plan does not branch on test scores.",
        test_previously_inspected=True,
        new_structural_exchanges=0,
        new_planned_updates=90840,
    )


def subcontext(ctx, rank):
    return dict(ctx, base=dict(ctx["base"], block_rank=rank, local_save_updates=50))


def validate_previous(ctx, source):
    cfg = source.config
    if (
        source.run_id != PREVIOUS_RUN
        or cfg["continuation_recipe"] != previous.settings()
        or cfg["base"] != ctx["base"]
        or cfg["fit_config"] != ctx["cfg"]
        or cfg["versions"] != matched.versions()
        or cfg["cap040_run"] != ctx["sources"]["cap040"].run_id
        or cfg["parent_run"] != ctx["parent"].run_id
    ):
        raise ValueError("Completed continuation source protocol or lineage differs")
    frozen = memory.required(source, "continuation_models_frozen")
    summary = memory.required(source, "experiment_complete")
    if (
        not summary["complete"]
        or not frozen["complete"]
        or len(frozen["models"]) != 9
        or {(r["seed"], r["arm"]) for r in frozen["models"]}
        != {(s, a) for s in SEEDS for a in previous.ARMS}
    ):
        raise ValueError(
            "All nine prior continuation models and completed audits are required"
        )
    for row in frozen["models"]:
        state = memory.required(source, row["key"])
        matched.complete(state, 3000)
        if (
            training.weight_sha(state["weights"]) != row["weights_sha"]
            or identity(engine.state_identity(state["optimizer"]))
            != row["optimizer_sha"]
            or (
                row["supervised"]
                and training.weight_sha({"head": state["head"]}) != row["head_sha"]
            )
            or row["targets_sha"] != identity(row["targets"])
            or row["rank"] != 4
            or row["total_training_updates"] != 13640
        ):
            raise ValueError("Prior endpoint identity changed: " + row["key"])
        if (
            state["sampled_batches_sha"] != row["phase_batch_sha"]
            or state["sampled_noise_sha"] != row["phase_noise_sha"]
        ):
            raise ValueError(
                "Prior frozen batch/noise hashes differ from the actual checkpoint"
            )
        root = next(
            r
            for r in ctx["roots"]
            if r["seed"] == row["seed"] and r["arm"] == row["starting_arm"]
        )
        if (
            row["starting_weights_sha"] != root["weights_sha"]
            or row["targets_sha"] != root["targets_sha"]
        ):
            raise ValueError(
                "Prior endpoint does not descend from its declared saved root"
            )
        tag = f"s{row['seed']}_{row['arm']}"
        saved = memory.required(source, "audit_" + tag)
        compact = memory.required(source, "compact_" + tag)
        if saved["identity"] != base.report_signature(
            row, ctx["test_stamp"], ctx["cfg"]
        ):
            raise ValueError("Prior audit uses a different model/test protocol")
        base.check_report(saved["report"], row, ctx["cfg"])
        sr = [
            r
            for r in summary["models"]
            if r["seed"] == row["seed"] and r["arm"] == row["arm"]
        ]
        if len(sr) != 1 or identity(sr[0]) != identity(saved["report"]):
            raise ValueError("Prior summary and audit differ")
        if (
            compact["image_ids"] != ctx["test_ids"]
            or compact["category_indices"] != ctx["cfg"]["category_indices"]
        ):
            raise ValueError("Prior compact audit cohort differs")
        for prefix, value in [("audit_", saved), ("compact_", compact)]:
            ctx["store"].save(prefix + tag, value)
        for key in [row["key"], "audit_" + tag, "compact_" + tag]:
            ctx["checkpoint_stamps"].append(matched.source_key_stamp(source, key))
        ctx["roots"].append(dict(row, source_name="previous"))
        print("Verified saved endpoint and reusable audit:", tag, flush=True)
        del state, compact, saved
    ctx["checkpoint_stamps"].extend(
        matched.source_key_stamp(source, k)
        for k in ["continuation_models_frozen", "experiment_complete"]
    )


def prepare(
    paths,
    output,
    local,
    cache_local,
    source_sha,
    previous_output,
    previous_test_output=None,
    require_drive=True,
):
    base.safe_destination(output, list(paths.values()) + [previous_output])
    ctx = previous.prepare(
        paths,
        Path(local) / "verification",
        local,
        cache_local,
        source_sha,
        previous_test_output,
        False,
    )
    old = ctx["store"]
    payload = dict(
        old.config,
        study="agf_prioritized_three_experiments",
        continuation_recipe=settings(),
        previous_run=PREVIOUS_RUN,
        maximum_planned_gradient_updates=90840,
        test_access="After independent three-seed freeze of each priority stage",
    )
    ctx["store"] = inputs.JournalStore(
        output, payload, Path(local) / "next_rescue", require_drive
    )
    base.install_drive_recovery(ctx["store"])
    ctx["recipe"] = settings()
    for seed in SEEDS:
        for arm in base.CONTROLS + ["cap040"]:
            for prefix in ["audit_", "compact_"]:
                value = old.load(f"{prefix}s{seed}_{arm}")
                if value is not None:
                    ctx["store"].save(f"{prefix}s{seed}_{arm}", value)
    source = matched.discover(
        previous_output, Path(local) / "read_previous", run=PREVIOUS_RUN
    )
    ctx["sources"]["previous"] = source
    ctx["stamps"] += memory.source_snapshot([source])
    validate_previous(ctx, source)
    d4 = memory.required(ctx["parent"], "mask_dictionary4")
    check_dictionary4(d4)
    if [r["category"] for r in d4["records"]] != ctx["cfg"]["category_indices"]:
        raise ValueError("Saved initializer category order differs")
    ctx["checkpoint_stamps"].append(
        matched.source_key_stamp(ctx["parent"], "mask_dictionary4")
    )
    ctx["store"].save(
        "all_models_frozen", dict(complete=True, models=ctx["cap040_rows"])
    )
    ctx["store"].write("protocol.json", payload)
    ctx["store"].write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    matched.assert_sources(ctx)
    return ctx


def check_dictionary4(value):
    D = value["D"]
    if (
        tuple(D.shape) != (300, 768)
        or not torch.isfinite(D).all()
        or value["sha"] != training.weight_sha({"D": D})
        or not torch.allclose(
            D.reshape(75, 4, 768) @ D.reshape(75, 4, 768).transpose(1, 2),
            torch.eye(4),
            atol=2e-5,
        )
    ):
        raise ValueError("Saved four-direction initializer is corrupt or nonorthogonal")


def orthogonal_fifth(gram, saved):
    if (
        gram.ndim != 2
        or gram.shape[0] != gram.shape[1]
        or saved.shape != (4, gram.shape[0])
        or not np.isfinite(gram).all()
        or not np.isfinite(saved).all()
        or not np.allclose(gram, gram.T, atol=1e-8)
        or not np.allclose(saved @ saved.T, np.eye(4), atol=2e-5)
    ):
        raise ValueError("Invalid Gram matrix or saved four axes")
    eig, V = np.linalg.eigh(gram)
    alignment = np.linalg.svd(saved @ V[:, -4:], compute_uv=False)
    if alignment.min() < 0.999:
        raise ValueError(
            "Recomputed TRAIN eigenspace disagrees with saved four directions"
        )
    if eig[-5] <= max(1e-12, float(eig[-1]) * 1e-12):
        raise ValueError("Fifth direction has no measured TRAIN energy")
    d = V[:, -5].copy()
    d -= saved.T @ (saved @ d)
    norm = np.linalg.norm(d)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Fifth direction is degenerate")
    return d / norm, eig, alignment


def dictionary5(ctx):
    store = ctx["store"]
    D4 = memory.required(ctx["parent"], "mask_dictionary4")["D"].reshape(75, 4, 768)
    old = store.load("mask_dictionary5")
    if old is not None:
        D = old["D"]
        if (
            tuple(D.shape) != (375, 768)
            or old["sha"] != training.weight_sha({"D": D})
            or not torch.equal(D.reshape(75, 5, 768)[:, :4], D4)
            or not torch.allclose(
                D.reshape(75, 5, 768) @ D.reshape(75, 5, 768).transpose(1, 2),
                torch.eye(5),
                atol=2e-5,
            )
        ):
            raise ValueError("Saved fifth-direction dictionary changed")
        return D
    pack = ctx["packs"]["train"]
    rows = []
    reports = []
    for slot, ci in enumerate(ctx["cfg"]["category_indices"]):
        fingerprint = identity(
            dict(
                D4=training.weight_sha({"D": D4[slot]}),
                train_ids=pack["ids"],
                normalization=identity(ctx["norm"]),
                positive_fraction=0.5,
                crowd_fraction=0.1,
            )
        )
        key = f"fifth_direction_{ci:02d}"
        record = store.load(key)
        if record is None:
            store.event("fifth_direction_start", category_index=ci)
            gram = np.zeros((768, 768), np.float64)
            count = 0
            images = 0
            positive = negative = 0.0
            for i, m in enumerate(pack["masks"]):
                take = (np.asarray(m["crowd"]) <= 0.1) & (
                    m["fraction"][:, ci].astype(np.float32) / 255 >= 0.5
                )
                if take.any():
                    x = np.asarray(pack["x"][i, take], np.float64)
                    gram += x.T @ x
                    count += len(x)
                    images += 1
            if count < 5:
                raise ValueError(
                    "Insufficient foreground patches for category " + str(ci)
                )
            d, eig, alignment = orthogonal_fifth(
                gram / count, D4[slot].numpy().astype(np.float64)
            )
            for i, m in enumerate(pack["masks"]):
                take = (np.asarray(m["crowd"]) <= 0.1) & (
                    m["fraction"][:, ci].astype(np.float32) / 255 >= 0.5
                )
                if take.any():
                    projection = np.asarray(pack["x"][i, take], np.float64) @ d
                    positive += float(np.square(projection[projection > 0]).sum())
                    negative += float(np.square(projection[projection < 0]).sum())
            if negative > positive:
                d = -d
            record = dict(
                fingerprint=fingerprint,
                direction=d.astype(np.float32),
                category_index=ci,
                patches=count,
                images=images,
                leading_eigenvalues=eig[-5:][::-1].tolist(),
                alignment=alignment.tolist(),
            )
            store.save(key, record)
        if (
            record["fingerprint"] != fingerprint
            or not np.isfinite(record["direction"]).all()
        ):
            raise ValueError("Saved fifth-direction selection differs")
        block = torch.cat([D4[slot], torch.as_tensor(record["direction"])[None]], 0)
        if not torch.allclose(block @ block.T, torch.eye(5), atol=2e-5):
            raise ValueError("Five axes are not orthonormal")
        rows.append(block)
        reports.append(record)
    D = torch.cat(rows, 0)
    store.save(
        "mask_dictionary5",
        dict(D=D, sha=training.weight_sha({"D": D}), reports=reports),
    )
    store.write("initialization5.json", base.width.json_safe(reports))
    return D


def root_for(ctx, seed, arm):
    rows = [r for r in ctx["roots"] if r["seed"] == seed and r["arm"] == arm]
    if len(rows) != 1:
        raise ValueError("Ambiguous saved root")
    return rows[0]


def source_state(ctx, row):
    state = memory.required(ctx["sources"][row["source_name"]], row["key"])
    matched.complete(state, 3000 if row["source_name"] == "previous" else 500)
    if (
        training.weight_sha(state["weights"]) != row["weights_sha"]
        or row.get("head_sha") is not None
        and training.weight_sha({"head": state["head"]}) != row["head_sha"]
        or row.get("optimizer_sha") is not None
        and identity(engine.state_identity(state["optimizer"])) != row["optimizer_sha"]
    ):
        raise ValueError("Saved root weights, head or optimizer changed")
    return state


def endpoint_row(ctx, seed, arm, rank, state, targets, ref, **extra):
    supervised = arm not in ["vanilla5", "longer_vanilla"]
    sub = subcontext(ctx, rank)
    matched.valid_weights(state["weights"], "vanilla", sub["base"])
    v = state["history"][-1]["validation"]
    if not supervised:
        model = core.new_model("vanilla", sub["base"], state["weights"], ctx["device"])
        with torch.inference_mode():
            v = core.validation(
                model, ctx["packs"]["select"], targets, ctx["cfg"], ctx["device"]
            )
        del model
    base.bound.validate_metrics(v)
    return dict(
        seed=seed,
        arm=arm,
        label=LABELS[arm],
        rank=rank,
        origin="new",
        targets=targets,
        targets_sha=identity(targets),
        weights_sha=training.weight_sha(state["weights"]),
        head_sha=training.weight_sha({"head": state["head"]}) if supervised else None,
        optimizer_sha=identity(engine.state_identity(state["optimizer"])),
        balance_sha=identity(engine.state_identity(state["balance"])),
        supervised=supervised,
        balanced=supervised,
        core_parameters=sum(v.numel() for v in state["weights"].values()),
        head_parameters=150 if supervised else 0,
        maximum_active_coordinates=16 * rank,
        reference_mse=ref,
        development=v,
        phase_batch_sha=state["sampled_batches_sha"],
        phase_noise_sha=state["sampled_noise_sha"],
        **extra,
    )


def run_width5(ctx):
    D = dictionary5(ctx)
    sub = subcontext(ctx, 5)
    rows = []
    s = ctx["store"]
    x = ctx["packs"]["train"]["x"].reshape(-1, 768)
    v = ctx["packs"]["select"]["x"].reshape(-1, 768)
    # Reuse the original width-three RNG advance so the epoch permutations
    # match the saved 4D warm-ups despite a different number of parameters.
    warm_rng_ctx = dict(ctx, base=dict(ctx["base"], block_rank=3))
    for seed in SEEDS:
        ref = root_for(ctx, seed, "cap040")["reference_mse"]
        paired = []
        mask, random = base.width.initialize(sub["base"], D, seed)
        for kind, initial in [("mask", mask), ("vanilla", random)]:
            arm = "mask5" if kind == "mask" else "vanilla5"
            key = f"phase2_{arm}_s{seed}"
            warm = fit_warm(
                s,
                f"warm5_{kind}_s{seed}",
                "vanilla",
                x,
                v,
                sub["base"],
                seed,
                sub["base"]["lrs"][0],
                ctx["device"],
                initial=initial,
                drive_every=250,
                pair_rng=base.width.paired_warm_rng(warm_rng_ctx, seed),
            )
            matched.complete(warm, 7640)
            oldwarm = memory.required(ctx["parent"], f"warm4_{kind}_s{seed}")
            if warm["sampled_batches_sha"] != oldwarm["sampled_batches_sha"]:
                raise ValueError(
                    "5D warm-up image/patch schedule differs from saved 4D control"
                )
            if warm["best"] is None:
                raise ValueError("Missing development-selected reconstruction warm-up")
            targets = matched.training_targets(
                sub, "vanilla", warm["best"], f"targets5_{kind}_s{seed}"
            )
            supervised = kind == "mask"
            p1 = engine.fit(
                sub,
                f"phase1_{arm}_s{seed}",
                warm["best"],
                targets,
                2500,
                71000 + seed,
                supervised,
                supervised,
                ref,
            )
            matched.complete(p1, 2500)
            p2 = engine.fit(
                sub,
                key,
                p1["weights"],
                targets,
                500,
                82000 + seed,
                supervised,
                supervised,
                ref,
                initial_head=p1["head"] if supervised else None,
                initial_balance=p1["balance"] if supervised else None,
            )
            matched.complete(p2, 500)
            for phase, oldkey in [
                (p1, f"phase1_cap040_s{seed}"),
                (p2, f"phase2_cap040_s{seed}"),
            ]:
                saved = memory.required(ctx["sources"]["cap040"], oldkey)
                for field in ["sampled_batches_sha", "sampled_noise_sha"]:
                    if phase[field] != saved[field]:
                        raise ValueError(
                            "5D guided batch/noise schedule differs from saved 4D: "
                            + field
                        )
            row = endpoint_row(
                ctx,
                seed,
                arm,
                5,
                p2,
                targets,
                ref,
                key=key,
                checkpoint_updates=500,
                total_training_updates=10640,
                warm_selected_update=warm["best_update"],
                new_gradient_updates=10640,
                warm_batch_sha=warm["sampled_batches_sha"],
                warm_noise_sha=warm["sampled_noise_sha"],
                phase1_batch_sha=p1["sampled_batches_sha"],
                phase1_noise_sha=p1["sampled_noise_sha"],
            )
            paired.append(row)
            del warm, oldwarm, p1, p2, saved
            gc.collect()
            torch.cuda.empty_cache()
        for field in [
            "warm_batch_sha",
            "warm_noise_sha",
            "phase1_batch_sha",
            "phase1_noise_sha",
            "phase_batch_sha",
            "phase_noise_sha",
        ]:
            if len({r[field] for r in paired}) != 1:
                raise ValueError("Matched width-five schedules differ: " + field)
        rows.extend(paired)
    return rows


def run_continuation(ctx, stage):
    rows = []
    for seed in SEEDS:
        paired = []
        for arm in STAGE_ARMS[stage]:
            root_arm = (
                "cap040"
                if stage == "pcgrad_both"
                else ("vanilla_continue" if arm == "longer_vanilla" else "ggf_continue")
            )
            root = root_for(ctx, seed, root_arm)
            st = source_state(ctx, root)
            supervised = arm != "longer_vanilla"
            key = f"{arm}_s{seed}"
            offset = 93000 if stage == "pcgrad_both" else 94000
            out = engine.fit(
                ctx,
                key,
                st["weights"],
                root["targets"],
                3000,
                offset + seed,
                supervised,
                supervised,
                root["reference_mse"],
                initial_head=st["head"] if supervised else None,
                initial_balance=st["balance"] if supervised else None,
                initial_optimizer=st["optimizer"],
                pcgrad=stage == "pcgrad_both",
                decoder_pcgrad=stage == "pcgrad_both",
            )
            matched.complete(out, 3000)
            if stage == "pcgrad_both":
                for saved_arm in previous.ARMS:
                    old = root_for(ctx, seed, saved_arm)
                    for field in ["phase_batch_sha", "phase_noise_sha"]:
                        statefield = (
                            "sampled_batches_sha"
                            if field == "phase_batch_sha"
                            else "sampled_noise_sha"
                        )
                        if out[statefield] != old[field]:
                            raise ValueError(
                                "Broader PCGrad does not match saved schedule/noise"
                            )
            row = endpoint_row(
                ctx,
                seed,
                arm,
                4,
                out,
                root["targets"],
                root["reference_mse"],
                key=key,
                checkpoint_updates=3000,
                total_training_updates=13640 if stage == "pcgrad_both" else 16640,
                new_gradient_updates=3000,
                starting_arm=root_arm,
                starting_weights_sha=root["weights_sha"],
            )
            paired.append(row)
            del st, out
            gc.collect()
            torch.cuda.empty_cache()
        for field in ["phase_batch_sha", "phase_noise_sha"]:
            if len({r[field] for r in paired}) != 1:
                raise ValueError("Matched continuation schedules differ")
        rows.extend(paired)
    return rows


def verify_freeze(ctx, stage):
    frozen = memory.required(ctx["store"], "frozen_" + stage)
    if (
        not frozen["complete"]
        or {(r["seed"], r["arm"]) for r in frozen["models"]}
        != {(s, a) for s in SEEDS for a in STAGE_ARMS[stage]}
        or len(frozen["models"]) != 3 * len(STAGE_ARMS[stage])
    ):
        raise ValueError("Freeze every seed and arm of this experiment before test")
    for row in frozen["models"]:
        st = memory.required(ctx["store"], row["key"])
        matched.complete(st, row["checkpoint_updates"])
        if (
            training.weight_sha(st["weights"]) != row["weights_sha"]
            or identity(engine.state_identity(st["optimizer"])) != row["optimizer_sha"]
            or identity(engine.state_identity(st["balance"])) != row["balance_sha"]
            or row["supervised"]
            and training.weight_sha({"head": st["head"]}) != row["head_sha"]
        ):
            raise ValueError("Frozen checkpoint changed")
    return frozen


def run_stage(ctx, stage):
    if stage not in PRIORITY:
        raise ValueError("Unknown experiment")
    index = PRIORITY.index(stage)
    for earlier in PRIORITY[:index]:
        if not memory.required(ctx["store"], "result_" + earlier)["complete"]:
            raise ValueError("Complete the higher-priority experiment first")
    ctx["packs"].pop("test", None)
    gc.collect()
    old = ctx["store"].load("frozen_" + stage)
    if old is None:
        rows = run_width5(ctx) if stage == "width5" else run_continuation(ctx, stage)
        ctx["store"].save(
            "frozen_" + stage,
            dict(complete=True, models=rows, stage=stage, test_features_opened=False),
        )
        ctx["store"].write(
            "frozen_" + stage + ".json",
            base.width.json_safe(dict(complete=True, models=rows)),
        )
    frozen = verify_freeze(ctx, stage)
    matched.assert_sources(ctx)
    base.load_test(ctx)
    return evaluate_stage(ctx, stage, frozen)


def audit(ctx, row):
    s = ctx["store"]
    tag = f"s{row['seed']}_{row['arm']}"
    sig = base.report_signature(row, ctx["test_stamp"], ctx["cfg"])
    old = s.load("audit_" + tag)
    if old is not None:
        if old["identity"] != sig:
            raise ValueError("Cached evaluation identity differs")
        base.check_report(old["report"], row, ctx["cfg"])
        memory.required(s, "compact_" + tag)
        return old["report"]
    st = memory.required(s, row["key"])
    matched.complete(st, row["checkpoint_updates"])
    if training.weight_sha(st["weights"]) != row["weights_sha"]:
        raise ValueError("Frozen inference weights changed")
    model = core.new_model(
        "vanilla", subcontext(ctx, row["rank"])["base"], st["weights"], ctx["device"]
    )
    del st
    s.event(
        "test_scoring_start",
        model=tag,
        images=len(ctx["test_ids"]),
        new_gradient_updates=0,
    )
    with torch.inference_mode(), matched.progress(ctx, model, tag):
        r = core.validation(
            model,
            ctx["packs"]["test"],
            row["targets"],
            ctx["cfg"],
            ctx["device"],
            with_arrays=True,
        )
        compact = base.study.compact_audit(
            r.pop("_audit"), matched.mapping(row["targets"]), ctx["cfg"]
        )
        dev = training.measure(
            model,
            ctx["packs"]["select"]["x"].reshape(-1, model.d),
            ctx["device"],
            ctx["cfg"]["batch"],
        )
    if any(
        abs(dev[k] - row["development"]["reconstruction"][k]) > 1e-5
        for k in ["mse", "ve"]
    ):
        raise ValueError("Frozen development reconstruction does not reproduce")
    r.update(
        {
            k: row[k]
            for k in [
                "seed",
                "arm",
                "label",
                "rank",
                "origin",
                "core_parameters",
                "head_parameters",
                "maximum_active_coordinates",
                "weights_sha",
                "targets_sha",
                "total_training_updates",
            ]
        }
    )
    r.update(
        legacy_native_eligible=bool(dev["mse"] <= row["reference_mse"] * 1.02),
        development_reconstruction=dev,
        common_native_MSE_limit=row["reference_mse"] * 1.02,
    )
    base.check_report(r, row, ctx["cfg"])
    s.save("compact_" + tag, compact)
    s.save("audit_" + tag, dict(identity=sig, report=r))
    s.write("reports/" + tag + ".json", base.width.json_safe(r))
    del model, compact
    gc.collect()
    torch.cuda.empty_cache()
    return r


def evaluate_stage(ctx, stage, frozen):
    s = ctx["store"]
    reports = []
    comparisons = []
    for seed in SEEDS:
        lookup = {
            r["arm"]: audit(ctx, r) for r in frozen["models"] if r["seed"] == seed
        }
        for arm in STAGE_CONTROLS[stage]:
            row = root_for(ctx, seed, arm)
            saved = memory.required(s, f"audit_s{seed}_{arm}")
            if saved["identity"] != base.report_signature(
                row, ctx["test_stamp"], ctx["cfg"]
            ):
                raise ValueError("Saved control evaluation changed")
            base.check_report(saved["report"], row, ctx["cfg"])
            lookup[arm] = saved["report"]
        reports.extend(lookup.values())
        for a, b in PAIRS[stage]:
            ta = f"s{seed}_{a}"
            tb = f"s{seed}_{b}"
            key = f"paired_{ta}_vs_{b}"
            sig = identity(
                dict(
                    a=memory.required(s, "audit_" + ta)["identity"],
                    b=memory.required(s, "audit_" + tb)["identity"],
                    bootstrap=ctx["cfg"]["bootstrap"],
                )
            )
            old = s.load(key)
            if old is None:
                s.event(
                    "paired_intervals_start",
                    experiment=stage,
                    seed=seed,
                    treatment=a,
                    control=b,
                )
                ci = base.study.paired(
                    memory.required(s, "compact_" + ta),
                    memory.required(s, "compact_" + tb),
                    ctx["cfg"],
                )
                old = dict(identity=sig, intervals=ci)
                s.save(key, old)
            if old["identity"] != sig:
                raise ValueError("Paired interval identity changed")
            ar, br = lookup[a], lookup[b]
            comparisons.append(
                dict(
                    seed=seed,
                    treatment=a,
                    control=b,
                    paired_intervals=old["intervals"],
                    capture_gain_pp=100 * (ar["macro_capture"] - br["macro_capture"]),
                    AP_gain_pp=100 * (ar["macro_ap"] - br["macro_ap"]),
                    VE_gain_pp=100
                    * (ar["reconstruction"]["ve"] - br["reconstruction"]["ve"]),
                    reconstruction_guard_passed=ar["legacy_native_eligible"],
                    all_three_improve=all(
                        ar[k] > br[k] for k in ["macro_capture", "macro_ap"]
                    )
                    and ar["reconstruction"]["ve"] > br["reconstruction"]["ve"],
                )
            )
    out = dict(
        complete=True,
        stage=stage,
        models=reports,
        comparisons=comparisons,
        controls_retrained=False,
        new_test_gradient_updates=0,
        new_structural_exchanges=0,
        test_previously_inspected=True,
        uncertainty="Conditional whole-image paired intervals, 200 draws; no seed-level or multiplicity correction.",
    )
    stage_tables(ctx, out)
    matched.assert_sources(ctx)
    s.save("result_" + stage, out)
    s.write(stage + "_summary.json", base.width.json_safe(out))
    print(
        "\nEXPERIMENT COMPLETE:",
        stage,
        "(all three seeds). Results are saved before the next experiment.\n",
        flush=True,
    )
    return out


def stage_tables(ctx, out):
    stage = out["stage"]
    s = ctx["store"]
    rows = []
    for r in out["models"]:
        rows.append(
            dict(
                seed=r["seed"],
                arm=r["arm"],
                capture_pct=100 * r["macro_capture"],
                patch_AP_pct=100 * r["macro_ap"],
                VE_pct=100 * r["reconstruction"]["ve"],
                MSE=r["reconstruction"]["mse"],
                native_eligible=r["legacy_native_eligible"],
            )
        )
    means = []
    for arm in STAGE_ARMS[stage] + STAGE_CONTROLS[stage]:
        rr = [r for r in rows if r["arm"] == arm]
        if len(rr) != 3:
            raise ValueError("Every reported arm requires all three seeds")
        means.append(
            dict(
                arm=arm,
                **{
                    k: float(np.mean([r[k] for r in rr]))
                    for k in ["capture_pct", "patch_AP_pct", "VE_pct"]
                },
                eligible_seeds=sum(r["native_eligible"] for r in rr),
            )
        )
    matched.csv_write(s.results / (stage + "_test_comparison.csv"), rows)
    matched.csv_write(s.results / (stage + "_means.csv"), means)
    categories = [
        dict(seed=r["seed"], arm=r["arm"], **c)
        for r in out["models"]
        for c in r["categories"]
    ]
    matched.csv_write(s.results / (stage + "_categories.csv"), categories)
    flat = []
    for r in out["comparisons"]:
        ci = r["paired_intervals"]
        flat.append(
            {k: v for k, v in r.items() if k != "paired_intervals"}
            | dict(
                capture_ci_low=100 * ci["capture_gain_ci95"][0],
                capture_ci_high=100 * ci["capture_gain_ci95"][1],
                AP_ci_low=100 * ci["ap_gain_ci95"][0],
                AP_ci_high=100 * ci["ap_gain_ci95"][1],
            )
        )
    matched.csv_write(s.results / (stage + "_paired.csv"), flat)
    plot_means(s, means, stage)
    print(json.dumps(means, indent=2), flush=True)


def plot_means(store, means, stage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {
        "mask5": "5D AGF-G",
        "vanilla5": "5D Vanilla BSF",
        "cap040": "4D AGF-G / cap 0.40",
        "vanilla4": "4D Vanilla BSF",
        "pcgrad_both": "4D AGF-G / encoder + decoder PCGrad",
        "ggf_continue": "4D AGF-G / ordinary continuation",
        "ggf_pcgrad": "4D AGF-G / encoder PCGrad",
        "vanilla_continue": "4D Vanilla BSF / continuation",
        "longer_guided": "4D AGF-G / longer continuation",
        "longer_vanilla": "4D Vanilla BSF / longer continuation",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True, layout="constrained")
    for ax, key, title in zip(
        axes,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Capture", "Patch AP", "Reconstruction VE"],
    ):
        for j, r in enumerate(means):
            ax.barh(
                j,
                r[key],
                height=0.64,
                color="#178a7a" if r["arm"] in STAGE_ARMS[stage] else "#7d90a8",
            )
            ax.text(
                max(r[key], 0) + 1,
                j,
                f"{r[key]:.2f}%",
                va="center",
                ha="left",
                fontsize=10,
            )
        ax.set_yticks(range(len(means)), [labels[r["arm"]] for r in means], fontsize=10)
        ax.set_xlim(
            min(0, min(r[key] for r in means) - 15),
            max(100, max(r[key] for r in means) + 10),
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("%")
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    titles = dict(
        width5="Five directions per group",
        pcgrad_both="Encoder + decoder PCGrad",
        longer="Longer ordinary continuation",
    )
    axes[0].invert_yaxis()
    fig.suptitle(titles[stage] + " / means across seeds 21, 22, 23", fontsize=13)
    folder = store.results / "figures"
    folder.mkdir(exist_ok=True)
    fig.savefig(folder / (stage + ".png"), dpi=200)
    fig.savefig(folder / (stage + ".pdf"))
    plt.close(fig)


def export(ctx, source_path):
    stages = {stage: ctx["store"].load("result_" + stage) for stage in PRIORITY}
    out = dict(
        complete=all(v is not None and v["complete"] for v in stages.values()),
        priority=PRIORITY,
        stages=stages,
        planned_new_gradient_updates=90840,
        controls_retrained=False,
        test_previously_inspected=True,
        new_structural_exchanges=0,
    )
    ctx["store"].write("summary.json", base.width.json_safe(out))
    if out["complete"]:
        ctx["store"].save("experiment_complete", out)
    path = matched.export(ctx, source_path)
    target = path.with_name("agf_next_" + ctx["store"].run_id[:12] + "_results.zip")
    path.replace(target)
    return target


failure = base.failure
