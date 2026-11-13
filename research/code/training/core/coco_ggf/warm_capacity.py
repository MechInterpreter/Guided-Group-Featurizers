"""Paired warm-start width/gating experiments from final ordinary 5D models.

No backbone extraction, new mask initializer, mapping fit, Exchanges or PCGrad.
Every branch starts independently from the same completed 5D checkpoint.
"""

import copy, csv, gc, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import training
from coco_baselines.storage import identity, sha_file
from coco_coordinated import run as memory
from . import five_continuation as five, cap_followup as cache, capacity_fit as engine
from . import five_exchanges as lineage

matched = five.matched
SEEDS = [21, 22, 23]
UPDATES = 3000
TOTAL = 16640
CONFIGURATIONS = {"6d16": (6, 16), "8d16": (8, 16), "5d24": (5, 24), "5d16": (5, 16)}
ARMS = [kind + "_" + name for name in CONFIGURATIONS for kind in ["agf", "vanilla"]]
SOURCE_ARMS = ["guided5_continue", "vanilla5_continue"]
MODEL_ORDER = ["W_dec", "W_enc", "b_enc"]
PARENT_RUN = lineage.PARENT_RUN
PARENT_SOURCE_SHA = lineage.PARENT_SOURCE_SHA


def settings():
    balance = copy.deepcopy(cache.bound.settings()["balance"])
    if (
        balance["bounds"]["capture"][1] != 0.4
        or balance["bounds"]["detection"][1] != 0.4
    ):
        raise ValueError("Expected the established cap-0.40 recipe")
    return dict(
        version=1,
        study="agf_5d_warm_start_capacity_sweep",
        seeds=SEEDS,
        configurations={
            k: dict(rank=r, active=k0) for k, (r, k0) in CONFIGURATIONS.items()
        },
        arms=ARMS,
        groups=1024,
        updates_per_arm=UPDATES,
        new_planned_updates=UPDATES * len(ARMS) * len(SEEDS),
        seed_offset=97000,
        balance=balance,
        endpoint="Final 3000-update endpoints, no best-checkpoint or test selection.",
        initialization="Width growth preserves five learned directions in every group. New unit decoder directions are seeded random vectors orthogonal to that group's learned span; their encoder columns and biases start at zero. No new TRAIN eigen-axis computation.",
        optimizer="Carry Adam moments and step counters; expand new-coordinate moments with zeros. Existing step counters are shared with new coordinates; there is no independent per-coordinate Adam age.",
        controller="Carry log weights, Adam, reference losses and adaptation count; shared-encoder controller only, cap 0.40.",
        budgets="Every new AGF/Vanilla arm gets the same 3000 additional image/noise updates. New 5D/top-16 continuations isolate added-training effects. Source references are reused without refitting.",
        gating="Raw group coefficient L2 norms, unchanged native Top-k; no division by group width.",
        readouts="Keep each source model's TRAIN-fixed category mappings, heat normalization, and guided classifier heads.",
        selection="All 24 endpoints freeze before any new test inference; report every endpoint, including failures.",
        test_previously_inspected=True,
        new_warmups=0,
        new_extractions=0,
        new_structural_exchanges=0,
        pcgrad=False,
        checkpoint_local_every=50,
        checkpoint_drive_every=250,
        scope="Warm-start capacity and gating study, not a from-scratch optimal-width sweep or independent test confirmation.",
    )


def arm_spec(arm):
    if arm not in ARMS:
        raise ValueError("Unknown capacity arm: " + str(arm))
    kind, name = arm.split("_", 1)
    rank, active = CONFIGURATIONS[name]
    return rank, active, kind == "agf"


def label(arm):
    if arm in SOURCE_ARMS:
        return "Saved " + (
            "AGF 5D / top-16"
            if arm == "guided5_continue"
            else "Vanilla BSF 5D / top-16"
        )
    r, k, sup = arm_spec(arm)
    return ("AGF" if sup else "Vanilla BSF") + f" {r}D / top-{k}"


def subcontext(ctx, rank, active):
    return dict(
        ctx,
        base=dict(ctx["base"], block_rank=rank, active_groups=active),
        recipe=settings(),
    )


def expand_tensor(value, key, groups, old_rank, new_rank):
    """Move each old slot to the SAME group; never append to flattened storage."""
    if new_rank < old_rank:
        raise ValueError("Width shrinking is not supported")
    if new_rank == old_rank:
        return value.detach().cpu().clone()
    shape = list(value.shape)
    if key == "W_enc":
        if shape[1] != groups * old_rank:
            raise ValueError("Encoder width mismatch")
        out = value.new_zeros((shape[0], groups, new_rank), device="cpu")
        out[:, :, :old_rank] = value.detach().cpu().reshape(shape[0], groups, old_rank)
        return out.reshape(shape[0], groups * new_rank)
    if key not in ["W_dec", "b_enc"] or shape[0] != groups * old_rank:
        raise ValueError("Decoder/bias width mismatch")
    out = value.new_zeros((groups, new_rank, *shape[1:]), device="cpu")
    out[:, :old_rank] = value.detach().cpu().reshape(groups, old_rank, *shape[1:])
    return out.reshape(groups * new_rank, *shape[1:])


def widen_weights(weights, groups, new_rank, seed):
    if set(weights) != set(MODEL_ORDER):
        raise ValueError("Unexpected native weights")
    old_rank = weights["W_dec"].shape[0] // groups
    d = weights["W_dec"].shape[1]
    if old_rank != 5 or new_rank not in [5, 6, 8] or d < new_rank:
        raise ValueError("Unsupported width growth")
    before = training.weight_sha(weights)
    out = {
        k: expand_tensor(v, k, groups, old_rank, new_rank) for k, v in weights.items()
    }
    if new_rank == old_rank:
        return out
    rng = np.random.default_rng(197000 + seed)
    old = weights["W_dec"].detach().cpu().reshape(groups, old_rank, d).double().numpy()
    dec = out["W_dec"].reshape(groups, new_rank, d)
    for group in range(groups):
        # Always generate three additions so the 6D addition is nested in 8D.
        q, _ = np.linalg.qr(old[group].T, mode="reduced")
        extra = []
        for candidate in rng.normal(size=(3, d)):
            v = candidate.copy()
            for _ in range(2):
                v -= q @ (q.T @ v)
                for previous in extra:
                    v -= previous * (previous @ v)
            norm = np.linalg.norm(v)
            if not np.isfinite(norm) or norm < 1e-8:
                raise ValueError("Degenerate width-growth axis")
            extra.append(v / norm)
        dec[group, old_rank:] = torch.as_tensor(
            np.stack(extra[: new_rank - old_rank]), dtype=dec.dtype
        )
    if training.weight_sha(weights) != before:
        raise ValueError("Growth mutated source weights")
    return out


def widen_optimizer(optimizer, weights, groups, new_rank, supervised):
    if len(optimizer["param_groups"]) != 1:
        raise ValueError("Unexpected source Adam groups")
    old_rank = weights["W_dec"].shape[0] // groups
    out = copy.deepcopy(optimizer)
    ids = out["param_groups"][0]["params"]
    if len(ids) != 3 + int(supervised):
        raise ValueError("Unexpected source Adam parameter order")
    for pid, key in zip(ids, MODEL_ORDER):
        state = out["state"].get(pid)
        if state is None:
            raise ValueError("Source Adam moments are missing")
        for name, value in list(state.items()):
            if torch.is_tensor(value) and value.ndim:
                if value.shape != weights[key].shape:
                    raise ValueError("Adam shape/order mismatch: " + key + "/" + name)
                state[name] = expand_tensor(value, key, groups, old_rank, new_rank)
            elif torch.is_tensor(value):
                state[name] = value.detach().cpu().clone()
    return out


def source_row(ctx, seed, supervised):
    arm = SOURCE_ARMS[0 if supervised else 1]
    rows = [r for r in ctx["roots"] if (r["seed"], r["arm"]) == (seed, arm)]
    if len(rows) != 1:
        raise ValueError("Ambiguous source root")
    return rows[0]


def source_state(ctx, row):
    st = memory.required(ctx["sources"]["continued5"], row["key"])
    matched.complete(st, 3000)
    five.check_state(ctx, row, st)
    return st


def initial_branch(ctx, row, st, rank):
    w = widen_weights(st["weights"], ctx["base"]["groups"], rank, row["seed"])
    opt = widen_optimizer(
        st["optimizer"], st["weights"], ctx["base"]["groups"], rank, row["supervised"]
    )
    return dict(
        weights=w,
        optimizer=opt,
        head=st["head"].clone(),
        balance=copy.deepcopy(st["balance"]),
    )


def validate_state(ctx, row, state):
    rank, active, sup = arm_spec(row["arm"])
    sub = subcontext(ctx, rank, active)
    matched.complete(state, UPDATES)
    matched.valid_weights(state["weights"], "vanilla", sub["base"])
    if (
        training.weight_sha(state["weights"]) != row["weights_sha"]
        or identity(row["targets"]) != row["targets_sha"]
        or identity(engine.state_identity(state["optimizer"])) != row["optimizer_sha"]
        or identity(engine.state_identity(state["balance"])) != row["balance_sha"]
        or sup
        and training.weight_sha({"head": state["head"]}) != row["head_sha"]
        or state["sampled_batches_sha"] != row["phase_batch_sha"]
        or state["sampled_noise_sha"] != row["phase_noise_sha"]
    ):
        raise ValueError("Capacity endpoint identity changed")
    if (
        row["rank"] != rank
        or row["active_groups"] != active
        or row["maximum_active_coordinates"] != rank * active
    ):
        raise ValueError("Endpoint gate/capacity metadata changed")
    if sup:
        weights = state["balance"]["log_weights"].exp()
        if not torch.isfinite(weights).all() or not (
            0.0125 - 1e-7 <= weights[0] <= 0.4 + 1e-7
            and 0.0025 - 1e-7 <= weights[1] <= 0.4 + 1e-7
        ):
            raise ValueError("Endpoint controller exceeded cap 0.40")
    elif state["balance"] is not None:
        raise ValueError("Vanilla controller must be absent")


def prepare(paths, output, local, cache_local, source_sha, require_drive=True):
    cache.safe_destination(output, paths.values())
    local = Path(local)
    parent = matched.discover(
        paths["continued5"], local / "read_continued5", run=PARENT_RUN
    )
    if (
        parent.config["source_sha"] != PARENT_SOURCE_SHA
        or parent.config["versions"] != matched.versions()
    ):
        raise ValueError("Completed ordinary 5D source or numerical versions differ")
    frozen = memory.required(parent, "frozen_continuation5")
    result = memory.required(parent, "result_continuation5")
    lineage.validate_parent_catalog(frozen, result)
    del result
    width = matched.discover(
        paths["width_balance"],
        local / "read_width_balance",
        run=parent.config["parent_run"],
    )
    original = matched.discover(
        paths["original"],
        local / "read_original",
        run=width.config["source_runs"]["original"],
    )
    strong = matched.discover(
        paths["strong"],
        local / "read_strong",
        run=width.config["source_runs"]["strong"],
    )
    if original.run_id != memory.EXPECTED_PARENT:
        raise ValueError("Unexpected activation lineage")
    cfg = copy.deepcopy(parent.config["fit_config"])
    cfg.update(evaluate_every=250, local_save_every=50)
    base = dict(parent.config["base"], block_rank=5, active_groups=16)
    provenance = json.loads((original.results / "provenance.json").read_text())
    norm = memory.required(original, "normalization")
    plan = memory.required(strong, "fresh_plan")
    stamp = memory.required(strong, "activation_identity_test")
    ids = plan["splits"]["test"]
    if (
        stamp != width.config["test_identity"]
        or stamp["ids"] != ids
        or len(ids) != 1500
        or len(set(ids)) != 1500
        or stamp["normalization_sha"] != identity(norm)
    ):
        raise ValueError("Unexpected cached test/normalization identity")
    splits = [set(provenance["splits"][k]) for k in ["train", "select", "confirm"]] + [
        set(ids)
    ]
    if any(splits[i] & splits[j] for i in range(len(splits)) for j in range(i)):
        raise ValueError("Split overlap")
    if [len(provenance["splits"][k]) for k in ["train", "select", "confirm"]] != [
        2000,
        400,
        400,
    ]:
        raise ValueError("Unexpected split sizes")
    roots = [r for r in frozen["models"] if r["arm"] in SOURCE_ARMS]
    if len(roots) != 6:
        raise ValueError("Missing guided/Vanilla ordinary roots")
    payload = dict(
        recipe=settings(),
        source_sha=source_sha,
        parent_run=PARENT_RUN,
        parent_source_sha=PARENT_SOURCE_SHA,
        versions=matched.versions(),
        base=base,
        fit_config=cfg,
        activation_run=original.run_id,
        strong_run=strong.run_id,
        normalization_sha=identity(norm),
        test_identity=stamp,
        roots=[
            dict(
                seed=r["seed"],
                arm=r["arm"],
                weights_sha=r["weights_sha"],
                targets_sha=r["targets_sha"],
                optimizer_sha=r["optimizer_sha"],
                balance_sha=r["balance_sha"],
            )
            for r in roots
        ],
    )
    store = inputs.JournalStore(output, payload, local / "rescue", require_drive)
    cache.install_drive_recovery(store)
    ctx = dict(
        store=store,
        sources=dict(
            continued5=parent, width_balance=width, original=original, strong=strong
        ),
        roots=roots,
        cfg=cfg,
        base=base,
        device="cuda",
        local=local,
        norm=norm,
        provenance=provenance,
        test_ids=ids,
        test_stamp=stamp,
        test_chunk=strong.config["config"]["extract_chunk"],
        stamps=memory.source_snapshot([parent, width, original, strong]),
        checkpoint_stamps=[],
        packs={},
        recipe=settings(),
    )
    for source, keys in [
        (parent, ["frozen_continuation5", "result_continuation5"]),
        (original, ["normalization"]),
        (strong, ["fresh_plan", "activation_identity_test"]),
    ]:
        ctx["checkpoint_stamps"] += [matched.source_key_stamp(source, k) for k in keys]
    ctx["stamps"].append(
        dict(
            path=str(original.results / "provenance.json"),
            sha256=sha_file(original.results / "provenance.json"),
        )
    )
    # Validate every source checkpoint before loading the large feature caches.
    for row in roots:
        state = source_state(ctx, row)
        matched.check_targets(row["targets"], cfg)
        ctx["checkpoint_stamps"].append(matched.source_key_stamp(parent, row["key"]))
        del state
        gc.collect()
    memory.memory_safe_metrics()
    for split in ["train", "select", "confirm"]:
        ctx["packs"][split] = cache.cached_split(ctx, split, cache_local)
    store.write("protocol.json", payload)
    store.write(
        "split_roles.json",
        dict(
            train=provenance["splits"]["train"],
            select=provenance["splits"]["select"],
            confirm=provenance["splits"]["confirm"],
            test=ids,
            test_features_opened=False,
            test_previously_inspected=True,
        ),
    )
    store.write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    matched.assert_sources(ctx)
    return ctx


class ScratchStore:
    """Disposable preflight progress never shares a production fit key."""

    def __init__(self):
        self.saved = {}

    def load(self, key):
        return copy.deepcopy(self.saved.get(key))

    def save(self, key, value, drive=True):
        self.saved[key] = copy.deepcopy(value)
        return value

    def write(self, *args, **kw):
        pass

    def event(self, *args, **kw):
        pass


@torch.no_grad()
def check_growth_predictions(ctx, st, grown, rank):
    old = core.new_model(
        "vanilla", subcontext(ctx, 5, 16)["base"], st["weights"], ctx["device"]
    )
    new = core.new_model(
        "vanilla", subcontext(ctx, rank, 16)["base"], grown["weights"], ctx["device"]
    )
    if [k for k, _ in new.named_parameters()] != MODEL_ORDER:
        raise ValueError("Model Adam order changed")
    x = torch.as_tensor(
        np.array(ctx["packs"]["train"]["x"][0], copy=True), device=ctx["device"]
    )
    a, z = old(x)
    b, zz = new(x)
    rel = float((a - b).norm() / a.norm().clamp_min(1e-8))
    if (
        rel > 1e-5
        or not torch.allclose(z, zz[:, :, :5], rtol=1e-5, atol=1e-5)
        or rank > 5
        and torch.count_nonzero(zz[:, :, 5:])
    ):
        raise ValueError(
            "Width-growth initialization changed source predictions beyond floating-point tolerance"
        )
    del old, new
    gc.collect()
    torch.cuda.empty_cache()
    return rel


def preflight(ctx):
    hardware = dict(gpu=torch.cuda.get_device_name(), versions=matched.versions())
    old = ctx["store"].load("capacity_preflight")
    if old is not None and old["hardware"] == hardware:
        pending = pending_updates(ctx)
        old = copy.deepcopy(old)
        old["pending_updates"] = pending
        old["estimated_remaining_gradient_hours"] = (
            sum(pending[a] * old["timings"][a]["seconds_per_update"] for a in ARMS)
            / 3600
        )
        ctx["store"].write("preflight.json", old)
        return old
    ts = torch.get_rng_state()
    cs = torch.cuda.get_rng_state_all()
    timings = {}
    try:
        # Both supervised/Vanilla paths are exercised at every requested gate.
        for arm in ARMS:
            rank, active, sup = arm_spec(arm)
            root = source_row(ctx, 21, sup)
            st = source_state(ctx, root)
            grown = initial_branch(ctx, root, st, rank)
            sub = subcontext(ctx, rank, active)
            sub["store"] = ScratchStore()
            relative = check_growth_predictions(ctx, st, grown, rank)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            out = engine.fit(
                sub,
                "disposable_" + arm,
                grown["weights"],
                root["targets"],
                10,
                97021,
                sup,
                sup,
                root["reference_mse"],
                initial_head=grown["head"] if sup else None,
                initial_balance=grown["balance"] if sup else None,
                initial_optimizer=grown["optimizer"],
            )
            probe = core.new_model(
                "vanilla", sub["base"], out["weights"], ctx["device"]
            )
            with torch.no_grad():
                z = probe.encode(
                    torch.as_tensor(
                        np.array(ctx["packs"]["train"]["x"][0], copy=True),
                        device=ctx["device"],
                    )
                )
                if (
                    z.shape[1:] != (1024, rank)
                    or (z.norm(dim=-1) > 1e-6).sum(-1).max() > active
                ):
                    raise ValueError("Capacity preflight gate mismatch")
            timings[arm] = dict(
                seconds_per_update=out["seconds"] / 10,
                elapsed_including_audits=time.perf_counter() - start,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
                growth_relative_error=relative,
                controller_adaptations=None
                if not sup
                else out["balance"]["adaptations"],
            )
            ctx["store"].event("capacity_preflight_arm", arm=arm, **timings[arm])
            del st, grown, out, probe, z, sub
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.set_rng_state(ts)
        torch.cuda.set_rng_state_all(cs)
    pending = pending_updates(ctx)
    out = dict(
        hardware=hardware,
        timings=timings,
        pending_updates=pending,
        estimated_remaining_gradient_hours=sum(
            pending[a] * timings[a]["seconds_per_update"] for a in ARMS
        )
        / 3600,
        note="Gradient time only. Development/test scoring, bootstrap intervals and verified Drive writes add time. Eight disposable 10-update fits do not change source or production fit RNGs.",
    )
    ctx["store"].save("capacity_preflight", out)
    ctx["store"].write("preflight.json", out)
    matched.assert_sources(ctx)
    return out


def pending_updates(ctx):
    pending = {}
    for arm in ARMS:
        count = 0
        for seed in SEEDS:
            state = ctx["store"].load(f"{arm}_s{seed}")
            count += UPDATES - (0 if state is None else state["updates"])
            del state
        pending[arm] = count
    return pending


def fit_branch(ctx, arm, seed, max_updates=None):
    rank, active, sup = arm_spec(arm)
    root = source_row(ctx, seed, sup)
    st = source_state(ctx, root)
    grown = initial_branch(ctx, root, st, rank)
    ctx["store"]._capacity_active_key = f"{arm}_s{seed}"
    out = engine.fit(
        subcontext(ctx, rank, active),
        f"{arm}_s{seed}",
        grown["weights"],
        root["targets"],
        UPDATES,
        97000 + seed,
        sup,
        sup,
        root["reference_mse"],
        initial_head=grown["head"] if sup else None,
        initial_balance=grown["balance"] if sup else None,
        initial_optimizer=grown["optimizer"],
        max_updates=max_updates,
    )
    ctx["store"]._capacity_active_key = None
    del st, grown
    gc.collect()
    return out


def endpoint_row(ctx, arm, seed, state):
    rank, active, sup = arm_spec(arm)
    root = source_row(ctx, seed, sup)
    sub = subcontext(ctx, rank, active)
    model = core.new_model("vanilla", sub["base"], state["weights"], ctx["device"])
    development = {}
    with torch.inference_mode():
        for split in ["select", "confirm"]:
            report = core.validation(
                model, ctx["packs"][split], root["targets"], ctx["cfg"], ctx["device"]
            )
            cache.bound.validate_metrics(report)
            development[split] = report
    del model
    gc.collect()
    torch.cuda.empty_cache()
    row = dict(
        seed=seed,
        arm=arm,
        label=label(arm),
        rank=rank,
        active_groups=active,
        groups=ctx["base"]["groups"],
        origin="independent_warm_start_from_final_ordinary_5D",
        key=f"{arm}_s{seed}",
        checkpoint_updates=UPDATES,
        targets=copy.deepcopy(root["targets"]),
        targets_sha=root["targets_sha"],
        reference_mse=root["reference_mse"],
        weights_sha=training.weight_sha(state["weights"]),
        optimizer_sha=identity(engine.state_identity(state["optimizer"])),
        balance_sha=identity(engine.state_identity(state["balance"])),
        head_sha=training.weight_sha({"head": state["head"]}) if sup else None,
        supervised=sup,
        balanced=sup,
        pcgrad=False,
        decoder_pcgrad=False,
        core_parameters=sum(v.numel() for v in state["weights"].values()),
        head_parameters=state["head"].numel() if sup else 0,
        maximum_active_coordinates=rank * active,
        total_training_updates=TOTAL,
        new_gradient_updates=UPDATES,
        starting_arm=root["arm"],
        starting_weights_sha=root["weights_sha"],
        starting_optimizer_sha=root["optimizer_sha"],
        starting_balance_sha=root["balance_sha"],
        development=development["select"],
        confirmation=development["confirm"],
        phase_batch_sha=state["sampled_batches_sha"],
        phase_noise_sha=state["sampled_noise_sha"],
        final_weights=None
        if not sup
        else state["balance"]["log_weights"].exp().tolist(),
    )
    validate_state(ctx, row, state)
    return row


def check_pairing(rows):
    for seed in SEEDS:
        rr = [r for r in rows if r["seed"] == seed]
        if {r["arm"] for r in rr} != set(ARMS):
            raise ValueError("Incomplete seed capacity catalog")
        for field in ["phase_batch_sha", "phase_noise_sha"]:
            if len({r[field] for r in rr}) != 1:
                raise ValueError("Capacity arms use different image/noise schedules")
        for sup in [True, False]:
            branch = [r for r in rr if r["supervised"] == sup]
            for field in [
                "starting_arm",
                "starting_weights_sha",
                "starting_optimizer_sha",
                "starting_balance_sha",
                "targets_sha",
            ]:
                if len({r[field] for r in branch}) != 1:
                    raise ValueError(
                        "Capacity branches differ in their starting checkpoint: "
                        + field
                    )


def run(ctx, source_path=None):
    if "test" in ctx["packs"]:
        raise ValueError("Test must remain closed while training")
    old = ctx["store"].load("capacity_frozen")
    if old is not None:
        return verify_freeze(ctx)
    rows = []
    for arm in ARMS:
        ctx["store"].event(
            "capacity_arm_start", arm=arm, seeds=SEEDS, updates_per_seed=UPDATES
        )
        for seed in SEEDS:
            out = fit_branch(ctx, arm, seed)
            matched.complete(out, UPDATES)
            row = endpoint_row(ctx, arm, seed, out)
            rows.append(row)
            ctx["store"].write(f"development/{arm}_s{seed}.json", row)
            ctx["store"].save(
                "development_endpoints", dict(complete=False, models=rows)
            )
            development_tables(ctx, rows)
            del out
            gc.collect()
            torch.cuda.empty_cache()
        if source_path is not None:
            export(ctx, source_path)
    check_pairing(rows)
    frozen = dict(
        complete=True,
        models=rows,
        test_features_opened=False,
        planned_new_gradient_updates=72000,
    )
    ctx["store"].save("capacity_frozen", frozen)
    ctx["store"].write("capacity_frozen.json", frozen)
    verify_freeze(ctx)
    matched.assert_sources(ctx)
    return frozen


def verify_freeze(ctx):
    frozen = memory.required(ctx["store"], "capacity_frozen")
    rows = frozen["models"]
    if (
        not frozen["complete"]
        or len(rows) != 24
        or {(r["seed"], r["arm"]) for r in rows}
        != {(s, a) for s in SEEDS for a in ARMS}
    ):
        raise ValueError("Freeze all 24 capacity endpoints before test")
    check_pairing(rows)
    for row in rows:
        state = memory.required(ctx["store"], row["key"])
        validate_state(ctx, row, state)
        root = source_row(ctx, row["seed"], row["supervised"])
        for field, other in [
            ("starting_weights_sha", "weights_sha"),
            ("starting_optimizer_sha", "optimizer_sha"),
            ("starting_balance_sha", "balance_sha"),
            ("targets_sha", "targets_sha"),
        ]:
            if row[field] != root[other]:
                raise ValueError("Frozen capacity lineage changed")
        del state
        gc.collect()
    return frozen


def load_test(ctx):
    verify_freeze(ctx)
    sub = dict(ctx, cap040_rows=[source_row(ctx, s, True) for s in SEEDS])
    s = ctx["store"]

    class Barrier:
        def load(self, key):
            if key == "all_models_frozen":
                return dict(complete=True, models=sub["cap040_rows"])
            return s.load(key)

        def event(self, *a, **kw):
            return s.event(*a, **kw)

    sub["store"] = Barrier()
    old_verify = cache.verify_test_freeze

    def verify(actual, frozen):
        if identity(frozen["models"]) != identity(sub["cap040_rows"]):
            raise ValueError("Root barrier changed")
        verify_freeze(ctx)

    cache.verify_test_freeze = verify
    try:
        return cache.load_test(sub)
    finally:
        cache.verify_test_freeze = old_verify


def audit_signature(ctx, row):
    return identity(
        dict(
            weights=row["weights_sha"],
            targets=row["targets_sha"],
            rank=row["rank"],
            active_groups=row["active_groups"],
            test=ctx["test_stamp"],
            cfg=ctx["cfg"],
        )
    )


def audit(ctx, row):
    s = ctx["store"]
    tag = f"s{row['seed']}_{row['arm']}"
    sig = audit_signature(ctx, row)
    old = s.load("audit_" + tag)
    if old is not None:
        if old["identity"] != sig:
            raise ValueError("Cached capacity test gate/model identity differs")
        cache.check_report(old["report"], row, ctx["cfg"])
        lineage.compact_matches(
            memory.required(s, "compact_" + tag),
            ctx["test_ids"],
            ctx["cfg"]["category_indices"],
        )
        return old["report"]
    state = memory.required(s, row["key"])
    validate_state(ctx, row, state)
    model = core.new_model(
        "vanilla",
        subcontext(ctx, row["rank"], row["active_groups"])["base"],
        state["weights"],
        ctx["device"],
    )
    del state
    s.event(
        "test_scoring_start",
        model=tag,
        images=1500,
        rank=row["rank"],
        active_groups=row["active_groups"],
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
        compact = cache.study.compact_audit(
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
        raise ValueError("Frozen development reconstruction differs")
    r.update(
        {
            k: row[k]
            for k in [
                "seed",
                "arm",
                "label",
                "rank",
                "active_groups",
                "groups",
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
    cache.check_report(r, row, ctx["cfg"])
    s.save("compact_" + tag, compact)
    s.save("audit_" + tag, dict(identity=sig, report=r))
    s.write("reports/" + tag + ".json", r)
    s.event(
        "test_scoring_complete",
        model=tag,
        capture_pct=100 * r["macro_capture"],
        AP_pct=100 * r["macro_ap"],
        VE_pct=100 * r["reconstruction"]["ve"],
    )
    del model, compact
    gc.collect()
    torch.cuda.empty_cache()
    return r


def reuse_source_audit(ctx, row):
    s = ctx["store"]
    source = ctx["sources"]["continued5"]
    tag = f"s{row['seed']}_{row['arm']}"
    a = memory.required(source, "audit_" + tag)
    compact = memory.required(source, "compact_" + tag)
    # Source cfg is used for its unchanged signature; training logging cadence
    # can differ without reinterpreting an already saved source audit.
    if a["identity"] != cache.report_signature(
        row, ctx["test_stamp"], source.config["fit_config"]
    ):
        raise ValueError("Saved source audit identity changed")
    cache.check_report(a["report"], row, ctx["cfg"])
    lineage.compact_matches(compact, ctx["test_ids"], ctx["cfg"]["category_indices"])
    s.save("audit_" + tag, a)
    s.save("compact_" + tag, compact)
    s.write("reports/" + tag + ".json", a["report"])
    ctx["checkpoint_stamps"] += [
        matched.source_key_stamp(source, k) for k in ["audit_" + tag, "compact_" + tag]
    ]
    return a["report"]


def evaluation_pairs():
    pairs = [("agf_" + name, "vanilla_" + name) for name in CONFIGURATIONS]
    pairs += [
        (kind + "_" + name, kind + "_5d16")
        for name in ["6d16", "8d16", "5d24"]
        for kind in ["agf", "vanilla"]
    ]
    pairs += [("agf_5d16", "guided5_continue"), ("vanilla_5d16", "vanilla5_continue")]
    return pairs


def evaluate(ctx):
    frozen = verify_freeze(ctx)
    matched.assert_sources(ctx)
    old = ctx["store"].load("capacity_result")
    if old is not None:
        if not old["complete"]:
            raise ValueError("Unexpected incomplete result")
        return old
    load_test(ctx)
    reports = []
    comparisons = []
    s = ctx["store"]
    for seed in SEEDS:
        lookup = {
            r["arm"]: audit(ctx, r) for r in frozen["models"] if r["seed"] == seed
        }
        for source_arm in SOURCE_ARMS:
            root = next(
                r for r in ctx["roots"] if (r["seed"], r["arm"]) == (seed, source_arm)
            )
            lookup[source_arm] = reuse_source_audit(ctx, root)
        reports.extend(lookup.values())
        for a, b in evaluation_pairs():
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
            saved = s.load(key)
            if saved is None:
                s.event("paired_intervals_start", seed=seed, treatment=a, control=b)
                ci = cache.study.paired(
                    memory.required(s, "compact_" + ta),
                    memory.required(s, "compact_" + tb),
                    ctx["cfg"],
                )
                saved = dict(identity=sig, intervals=ci)
                s.save(key, saved)
            if saved["identity"] != sig:
                raise ValueError("Paired interval identities changed")
            ar, br = lookup[a], lookup[b]
            comparisons.append(
                dict(
                    seed=seed,
                    treatment=a,
                    control=b,
                    capture_gain_pp=100 * (ar["macro_capture"] - br["macro_capture"]),
                    AP_gain_pp=100 * (ar["macro_ap"] - br["macro_ap"]),
                    VE_gain_pp=100
                    * (ar["reconstruction"]["ve"] - br["reconstruction"]["ve"]),
                    native_eligible=ar["legacy_native_eligible"],
                    all_three_improve=bool(
                        ar["macro_capture"] > br["macro_capture"]
                        and ar["macro_ap"] > br["macro_ap"]
                        and ar["reconstruction"]["ve"] > br["reconstruction"]["ve"]
                    ),
                    paired_intervals=saved["intervals"],
                )
            )
    result = dict(
        complete=True,
        models=reports,
        comparisons=comparisons,
        seeds=SEEDS,
        new_planned_gradient_updates=72000,
        new_extractions=0,
        new_warmups=0,
        new_structural_exchanges=0,
        saved_controls_retrained=False,
        new_budget_matched_controls_fitted=True,
        all_models_frozen_before_test=True,
        test_previously_inspected=True,
        scope=settings()["scope"],
        uncertainty="Conditional whole-image paired intervals; no seed-level, researcher-selection or multiple-comparison adjustment.",
    )
    tables(ctx, reports, "test", comparisons)
    s.save("capacity_result", result)
    s.write("test_summary.json", result)
    matched.assert_sources(ctx)
    return result


def metric_row(r, v):
    return dict(
        seed=r["seed"],
        arm=r["arm"],
        label=label(r["arm"]),
        rank=r["rank"],
        active_groups=r.get("active_groups", 16),
        total_training_updates=r["total_training_updates"],
        core_parameters=r["core_parameters"],
        maximum_active_coordinates=r["maximum_active_coordinates"],
        capture_pct=100 * v["macro_capture"],
        patch_AP_pct=100 * v["macro_ap"],
        VE_pct=100 * v["reconstruction"]["ve"],
        MSE=v["reconstruction"]["mse"],
        native_eligible=bool(
            r["development"]["reconstruction"]["mse"] <= r["reference_mse"] * 1.02
        )
        if "reference_mse" in r
        else v["legacy_native_eligible"],
    )


def development_tables(ctx, rows):
    for split in ["select", "confirm"]:
        rr = []
        for row in rows:
            rr.append(
                metric_row(
                    row, row["development" if split == "select" else "confirmation"]
                )
            )
        # Reference roots have development metrics; confirmation metrics are incomplete.
        if split == "select":
            rr += [metric_row(r, r["development"]) for r in ctx["roots"]]
        write_tables(ctx, rr, "development_" + split)


def tables(ctx, reports, prefix, comparisons):
    rr = [metric_row(r, r) for r in reports]
    means = write_tables(ctx, rr, prefix)
    matched.csv_write(
        ctx["store"].results / (prefix + "_categories.csv"),
        [
            dict(seed=r["seed"], arm=r["arm"], **c)
            for r in reports
            for c in r["categories"]
        ],
    )
    flat = []
    for r in comparisons:
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
    matched.csv_write(ctx["store"].results / (prefix + "_paired.csv"), flat)
    plot_means(ctx, means, prefix)


def write_tables(ctx, rows, prefix):
    matched.csv_write(ctx["store"].results / (prefix + "_comparison.csv"), rows)
    means = []
    for arm in ARMS + SOURCE_ARMS:
        rr = [r for r in rows if r["arm"] == arm]
        if len(rr) == 3:
            means.append(
                dict(
                    arm=arm,
                    label=label(arm),
                    **{
                        k: float(np.mean([r[k] for r in rr]))
                        for k in ["capture_pct", "patch_AP_pct", "VE_pct"]
                    },
                    eligible_seeds=sum(r["native_eligible"] for r in rr),
                )
            )
    path = ctx["store"].results / (prefix + "_means.csv")
    if means:
        matched.csv_write(path, means)
    else:
        # A progress table can have fewer than three completed seeds.
        with path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(
                stream,
                fieldnames=[
                    "arm",
                    "label",
                    "capture_pct",
                    "patch_AP_pct",
                    "VE_pct",
                    "eligible_seeds",
                ],
            ).writeheader()
    print(
        prefix + " means (only complete three-seed arms):",
        json.dumps(means),
        flush=True,
    )
    return means


def plot_means(ctx, means, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.6), sharey=True, layout="constrained")
    for ax, key, title in zip(
        axes,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Capture", "Patch AP", "Reconstruction VE"],
    ):
        for i, r in enumerate(means):
            color = "#138778" if r["arm"].startswith("agf_") else "#879fb0"
            if r["arm"] in SOURCE_ARMS:
                color = "#c4bcc8"
            ax.barh(i, r[key], height=0.67, color=color)
            ax.text(r[key] + 1, i, f"{r[key]:.2f}%", va="center", fontsize=10)
        ax.set_yticks(range(len(means)), [r["label"] for r in means], fontsize=10)
        ax.set_xlim(0, 105)
        ax.set_title(title)
        ax.set_xlabel("%")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Warm-start capacity study / three seeds / previously inspected test")
    folder = ctx["store"].results / "figures"
    folder.mkdir(exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(folder / (prefix + "_means." + ext), dpi=200)
    plt.close(fig)


def flush_active_checkpoint(ctx):
    """Force the newest COMPLETE local checkpoint to Drive on graceful stop."""
    active = getattr(ctx["store"], "_capacity_active_key", None)
    if active is None:
        return
    saved = ctx["store"].load(active)
    if saved is None:
        return
    save = getattr(
        ctx["store"], "_width_drive_recovery_original_save", ctx["store"].save
    )
    save(active, saved, drive=True)
    print(
        "Verified last complete checkpoint on Drive:",
        active,
        saved["updates"],
        "/",
        saved["target"],
        flush=True,
    )


def export(ctx, source_path):
    result = ctx["store"].load("capacity_result")
    frozen = ctx["store"].load("capacity_frozen")
    progress = ctx["store"].load("development_endpoints")
    out = dict(
        complete=result is not None and result["complete"],
        recipe=settings(),
        endpoints=None if progress is None else progress["models"],
        all_endpoints_frozen=frozen is not None,
        result=result,
    )
    ctx["store"].write("summary.json", out)
    path = matched.export(ctx, source_path)
    target = path.with_name("agf_capacity_" + ctx["store"].run_id[:12] + "_results.zip")
    path.replace(target)
    return target


def failure(ctx, stage, exc, source_path):
    try:
        flush_active_checkpoint(ctx)
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        print("Partial evidence:", export(ctx, source_path), flush=True)
    except BaseException as err:
        print(
            "Partial export/Drive flush failed:",
            type(err).__name__,
            str(err),
            flush=True,
        )
