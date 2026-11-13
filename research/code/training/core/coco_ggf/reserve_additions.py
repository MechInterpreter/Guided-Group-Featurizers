"""Short development-only copy-addition screen, preceding the capacity sweep."""

import copy, gc, json, time, traceback
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import (
    warm_capacity as capacity,
    capacity_fit as engine,
    owner_routing as coverage,
)

SEEDS = [21, 22, 23]
ARMS = ["targeted", "random", "ordinary", "vanilla_ordinary"]
UPDATES = 500
LABELS = dict(
    targeted="AGF: targeted reserve copies",
    random="AGF: random reserve copies",
    ordinary="AGF: unchanged budget control",
    vanilla_ordinary="Vanilla BSF: budget control",
)


def settings():
    return dict(
        version=1,
        study="agf_reserved_concept_additions_prescreen",
        seeds=SEEDS,
        arms=ARMS,
        updates_per_arm=UPDATES,
        planned_new_updates=UPDATES * len(ARMS) * len(SEEDS),
        concept_groups=75,
        groups=1024,
        source_rank=5,
        padded_rank=6,
        active_groups=16,
        screening_images=512,
        training_seed=107011,
        profile_chunk=25,
        fit_seed_offset=111000,
        balance=capacity.settings()["balance"],
        operation="COPY one full learned encoder column, bias and decoder row from groups 75..1023 into a new sixth concept slot. Donors and all five original slots remain in place. New-coordinate Adam moments start at zero.",
        selection="TRAIN-only, largest additive object-contrast residual reduction among reserve directions with nonzero foreground raw activity. Uniform random addition from the same legal category pool; no development/test choice.",
        receivers="Initialized concept slots 0..74 in saved category order. Keep original TRAIN-selected capture/detection readouts and heads; log off-slot exceptions rather than silently retarget them.",
        padding="Native rectangular 6D storage; reserve sixth slots are exactly-zero, zero-moment padding. They cannot learn under this objective and are checked at every inspected checkpoint. Effective widths are 6 for concept slots, 5 for reserves.",
        preservation="No original slot is removed or reassigned. During the 500 updates, existing and added directions may train. Original directions are not numerically frozen.",
        controls="Targeted/random have identical usable capacity, gate, root, Adam/head/controller, image/noise streams and update budget. Ordinary AGF/Vanilla are budget controls, not addition-capacity controls.",
        gating="Unchanged raw group L2 Top-16, no width normalization; maximum active slots may rise from 80 to 96.",
        development="Initial and final reports on selection and confirmation; primary original readouts plus separately labelled initialized-slot diagnostics. No test features or intervals.",
        endpoint="Final 500 updates; retain all outcomes. No automatic adoption or search based on development.",
        no_warmup=True,
        no_backbone_extraction=True,
        no_removal_graph=True,
        checkpoint_local_every=50,
        checkpoint_drive_every=250,
    )


def supervised(arm):
    if arm not in ARMS:
        raise ValueError("Unknown reserve arm")
    return arm != "vanilla_ordinary"


def prepare(base, output, source_sha, require_drive=True):
    capacity.cache.safe_destination(
        output, [s.output for s in base["sources"].values()] + [base["store"].output]
    )
    payload = dict(
        capacity_context=base["store"].config, recipe=settings(), source_sha=source_sha
    )
    s = inputs.JournalStore(
        output, payload, base["local"] / "reserve_rescue", require_drive
    )
    capacity.cache.install_drive_recovery(s)
    ctx = dict(
        base,
        store=s,
        cfg=dict(base["cfg"], evaluate_every=250, local_save_every=50),
        packs={k: v for k, v in base["packs"].items() if k != "test"},
        recipe=settings(),
    )
    if ctx["base"]["groups"] != 1024 or len(ctx["cfg"]["category_indices"]) != 75:
        raise ValueError("Require the established 75-category / 1024-group source")
    for row in ctx["roots"]:
        if [t["category_index"] for t in row["targets"]] != ctx["cfg"][
            "category_indices"
        ]:
            raise ValueError("Initialized slot category order differs")
    s.write("protocol.json", payload)
    s.write(
        "scope.json",
        dict(
            test_features_opened=False,
            development_only=True,
            primary_readouts_unchanged=True,
            screen_does_not_change_later_capacity_roots=True,
        ),
    )
    return ctx


def slot_targets(targets):
    return [dict(t, contrast_group=i, detector_group=i) for i, t in enumerate(targets)]


def screen_pack(ctx):
    if "test" in ctx["packs"]:
        raise ValueError("Reserve screen must not access test features")
    old = ctx["store"].load("reserve_screen_indices")
    if old is None:
        allocation = dict(
            coverage.settings(),
            assignment_images=settings()["screening_images"],
            training_seed=settings()["training_seed"],
        )
        ix, evidence = coverage.assignment_indices(
            ctx["packs"]["train"],
            ctx["cfg"]["category_indices"],
            allocation,
            ctx["cfg"],
        )
        old = dict(
            indices=np.asarray(ix),
            ids=[ctx["packs"]["train"]["ids"][int(i)] for i in ix],
            evidence=evidence,
        )
        ctx["store"].save("reserve_screen_indices", old)
        ctx["store"].write("screen_coverage.json", evidence)
    p = ctx["packs"]["train"]
    ix = np.asarray(old["indices"])
    if old["ids"] != [p["ids"][int(i)] for i in ix]:
        raise ValueError("TRAIN screen order changed")
    return dict(
        x=np.array(p["x"][ix], copy=True),
        ids=old["ids"],
        masks=[p["masks"][int(i)] for i in ix],
    )


@torch.no_grad()
def profile_piece(model, pack, lo, hi, targets, cfg, concept_count):
    C = len(targets)
    start = 5 * concept_count
    A = model.W_dec.shape[0] - start
    out = dict(
        gain=np.zeros((C, A)),
        foreground=np.zeros((C, A)),
        background=np.zeros((C, A)),
        energy=np.zeros(C),
        counts=np.zeros(C, dtype=np.int64),
    )
    D = model.W_dec.detach()
    reserve = D[start:]
    dn = reserve.square().sum(1)
    lookup = {t["category_index"]: i for i, t in enumerate(targets)}
    for i in range(lo, hi):
        x = torch.as_tensor(np.array(pack["x"][i], copy=True), device=D.device)
        raw = x @ model.W_enc + model.b_enc
        z = model.encode(x).reshape(len(x), -1)
        weights, _, cats = core.object_weights([pack["masks"][i]], list(lookup), cfg)
        if not len(cats):
            continue
        wt = torch.as_tensor(weights, device=D.device)
        dx = wt @ x
        da = wt @ raw[:, start:]
        dz = wt @ z
        slots = torch.tensor([lookup[int(c)] for c in cats], device=D.device)
        ids = slots[:, None] * 5 + torch.arange(5, device=D.device)[None]
        prediction = torch.einsum("nr,nrd->nd", dz.gather(1, ids), D[ids])
        residual = dx - prediction
        gains = 2 * da * (residual @ reserve.T) - da.square() * dn
        fw = wt.clamp_min(0)
        bw = (-wt).clamp_min(0)
        fw = fw / fw.sum(1, keepdim=True).clamp_min(1e-12)
        bw = bw / bw.sum(1, keepdim=True).clamp_min(1e-12)
        fg = fw @ raw[:, start:].abs()
        bg = bw @ raw[:, start:].abs()
        for row, c in enumerate(cats):
            k = lookup[int(c)]
            out["gain"][k] += gains[row].double().cpu().numpy()
            out["foreground"][k] += fg[row].double().cpu().numpy()
            out["background"][k] += bg[row].double().cpu().numpy()
            out["energy"][k] += float(dx[row].square().sum())
            out["counts"][k] += 1
    return out


def choose_directions(profile, seed, concept_count):
    if (profile["counts"] <= 0).any() or (profile["energy"] <= 0).any():
        raise ValueError("Missing TRAIN category contrast evidence")
    gain = profile["gain"] / profile["energy"][:, None]
    fg = profile["foreground"] / profile["counts"][:, None]
    bg = profile["background"] / profile["counts"][:, None]
    if not np.isfinite(gain).all() or not np.isfinite(fg).all():
        raise ValueError("Nonfinite reserve profiles")
    rng = np.random.default_rng(211000 + seed)
    targeted = []
    random = []
    records = []
    for slot in range(concept_count):
        pool = np.flatnonzero(fg[slot] > 1e-8)
        if not len(pool):
            raise ValueError(
                "No foreground-responsive reserve direction for slot " + str(slot)
            )
        a = int(pool[np.argmax(gain[slot, pool])])
        b = int(rng.choice(pool))
        targeted.append(5 * concept_count + a)
        random.append(5 * concept_count + b)
        records.append(
            dict(
                slot=slot,
                eligible_pool=len(pool),
                targeted_atom=targeted[-1],
                random_atom=random[-1],
                targeted_predicted_capture_gain=float(gain[slot, a]),
                random_predicted_capture_gain=float(gain[slot, b]),
                targeted_foreground_activity=float(fg[slot, a]),
                targeted_comparison_activity=float(bg[slot, a]),
                targeted_predicted_positive=bool(gain[slot, a] > 0),
            )
        )
    return dict(
        targeted=targeted,
        random=random,
        records=records,
        note="Raw/additive approximation before actual gate changes. A largest score can still be nonpositive; all such cases remain reported.",
    )


def selection_for(ctx, seed):
    key = f"reserve_selection_s{seed}"
    old = ctx["store"].load(key)
    root = capacity.source_row(ctx, seed, True)
    pack = screen_pack(ctx)
    sig = identity(
        dict(
            root=root["weights_sha"],
            ids=pack["ids"],
            targets=slot_targets(root["targets"]),
            cfg=ctx["cfg"],
            recipe=settings(),
        )
    )
    if old is not None:
        if old["signature"] != sig:
            raise ValueError("Reserve donor selection identity changed")
        return old
    state = capacity.source_state(ctx, root)
    m = core.new_model(
        "vanilla",
        capacity.subcontext(ctx, 5, 16)["base"],
        state["weights"],
        ctx["device"],
    )
    del state
    C = len(root["targets"])
    A = ctx["base"]["groups"] * 5 - C * 5
    progress_key = f"reserve_profile_s{seed}"
    progress = ctx["store"].load(progress_key)
    if progress is None:
        progress = dict(
            signature=sig,
            next_offset=0,
            stats=dict(
                gain=np.zeros((C, A)),
                foreground=np.zeros((C, A)),
                background=np.zeros((C, A)),
                energy=np.zeros(C),
                counts=np.zeros(C, dtype=np.int64),
            ),
        )
    if progress["signature"] != sig:
        raise ValueError("Resumed TRAIN profile changed")
    for lo in range(
        progress["next_offset"], len(pack["ids"]), settings()["profile_chunk"]
    ):
        hi = min(lo + settings()["profile_chunk"], len(pack["ids"]))
        piece = profile_piece(m, pack, lo, hi, root["targets"], ctx["cfg"], C)
        # Commit only whole profile chunks. A failed unit replays at most 25 images.
        progress = dict(
            signature=sig,
            next_offset=hi,
            stats={k: progress["stats"][k] + piece[k] for k in piece},
        )
        ctx["store"].save(progress_key, progress)
        ctx["store"].event(
            "reserve_profile", seed=seed, done=hi, total=len(pack["ids"])
        )
    chosen = choose_directions(progress["stats"], seed, C)
    chosen.update(signature=sig, seed=seed, root_sha=root["weights_sha"])
    for t, r in zip(root["targets"], chosen["records"]):
        r.update(
            category=t["category"],
            category_index=t["category_index"],
            original_capture_group=t["contrast_group"],
            original_detector_group=t["detector_group"],
        )
    ctx["store"].save(key, chosen)
    ctx["store"].write(f"donors/seed{seed}.json", chosen)
    del m, pack
    gc.collect()
    torch.cuda.empty_cache()
    return chosen


def copy_additions(weights, groups, donors, concept_count):
    if len(donors) != concept_count:
        raise ValueError("One donor per initialized concept slot is required")
    before = training.weight_sha(weights)
    out = {k: capacity.expand_tensor(v, k, groups, 5, 6) for k, v in weights.items()}
    for slot, atom in enumerate(donors):
        if type(atom) is not int or not 5 * concept_count <= atom < 5 * groups:
            raise ValueError("Donor must come from a reserve group")
        out["W_dec"][slot * 6 + 5] = weights["W_dec"][atom]
        out["W_enc"][:, slot * 6 + 5] = weights["W_enc"][:, atom]
        out["b_enc"][slot * 6 + 5] = weights["b_enc"][atom]
    if training.weight_sha(weights) != before:
        raise ValueError("Copying mutated source weights")
    return out


def assert_padding(weights, groups, concept_count, has_additions):
    start = concept_count if has_additions else 0
    if (
        torch.count_nonzero(weights["W_dec"].reshape(groups, 6, -1)[start:, 5:])
        or torch.count_nonzero(weights["W_enc"].reshape(-1, groups, 6)[:, start:, 5:])
        or torch.count_nonzero(weights["b_enc"].reshape(groups, 6)[start:, 5:])
    ):
        raise ValueError("Disabled padding coordinates changed")


def assert_preserved_start(original, edited, groups):
    for key in ["W_dec", "b_enc"]:
        a = original[key].reshape(groups, 5, *original[key].shape[1:])
        b = edited[key].reshape(groups, 6, *original[key].shape[1:])
        if not torch.equal(a, b[:, :5]):
            raise ValueError("An existing direction/bias was altered during addition")
    if not torch.equal(
        original["W_enc"].reshape(-1, groups, 5),
        edited["W_enc"].reshape(-1, groups, 6)[:, :, :5],
    ):
        raise ValueError("An existing encoder coordinate was altered during addition")


def branch_start(ctx, arm, seed):
    sup = supervised(arm)
    root = capacity.source_row(ctx, seed, sup)
    source = capacity.source_state(ctx, root)
    C = len(root["targets"])
    G = ctx["base"]["groups"]
    donors = []
    if arm in ["targeted", "random"]:
        chosen = selection_for(ctx, seed)
        donors = chosen[arm]
        weights = copy_additions(source["weights"], G, donors, C)
    else:
        weights = {
            k: capacity.expand_tensor(v, k, G, 5, 6)
            for k, v in source["weights"].items()
        }
    assert_preserved_start(source["weights"], weights, G)
    assert_padding(weights, G, C, bool(donors))
    opt = capacity.widen_optimizer(source["optimizer"], source["weights"], G, 6, sup)
    result = dict(
        weights=weights,
        optimizer=opt,
        head=source["head"].clone(),
        balance=copy.deepcopy(source["balance"]),
        donors=donors,
    )
    del source
    gc.collect()
    return root, result


def fit_context(ctx):
    return dict(ctx, base=dict(ctx["base"], block_rank=6), recipe=settings())


@torch.no_grad()
def reports(ctx, weights, targets):
    model = core.new_model("vanilla", fit_context(ctx)["base"], weights, ctx["device"])
    out = {}
    for split in ["select", "confirm"]:
        r = core.validation(
            model,
            ctx["packs"][split],
            targets,
            ctx["cfg"],
            ctx["device"],
            with_arrays=True,
        )
        diagnostic = core.category_scores(
            r.pop("_audit"), slot_targets(targets), ctx["cfg"]
        )
        capacity.cache.bound.validate_metrics(r)
        out[split] = dict(primary=r, fixed_category_slot_diagnostic=diagnostic)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


def initial_report(ctx, arm, seed, root, start):
    key = f"reserve_initial_{arm}_s{seed}"
    sig = identity(
        dict(
            weights=training.weight_sha(start["weights"]),
            targets=root["targets_sha"],
            cfg=ctx["cfg"],
            ids={k: ctx["packs"][k]["ids"] for k in ["select", "confirm"]},
        )
    )
    old = ctx["store"].load(key)
    if old is not None:
        if old["signature"] != sig:
            raise ValueError("Initial addition scoring identity changed")
        return old
    old = dict(
        signature=sig,
        seed=seed,
        arm=arm,
        reports=reports(ctx, start["weights"], root["targets"]),
    )
    ctx["store"].save(key, old)
    ctx["store"].write(f"initial/{arm}_s{seed}.json", old)
    return old


def fit_branch(ctx, arm, seed, max_updates=None):
    root, start = branch_start(ctx, arm, seed)
    sup = supervised(arm)
    key = f"reserve_{arm}_s{seed}"
    ctx["store"]._capacity_active_key = key
    result = engine.fit(
        fit_context(ctx),
        key,
        start["weights"],
        root["targets"],
        UPDATES,
        111000 + seed,
        sup,
        sup,
        root["reference_mse"],
        initial_head=start["head"] if sup else None,
        initial_balance=start["balance"] if sup else None,
        initial_optimizer=start["optimizer"],
        max_updates=max_updates,
    )
    assert_padding(
        result["weights"],
        ctx["base"]["groups"],
        len(root["targets"]),
        arm in ["targeted", "random"],
    )
    ctx["store"]._capacity_active_key = None
    del start
    gc.collect()
    return root, result


def endpoint(ctx, arm, seed, root, state):
    scored = reports(ctx, state["weights"], root["targets"])
    C = len(root["targets"])
    G = ctx["base"]["groups"]
    d = state["weights"]["W_dec"].shape[1]
    return dict(
        seed=seed,
        arm=arm,
        label=LABELS[arm],
        key=f"reserve_{arm}_s{seed}",
        checkpoint_updates=UPDATES,
        starting_weights_sha=root["weights_sha"],
        starting_optimizer_sha=root["optimizer_sha"],
        starting_balance_sha=root["balance_sha"],
        targets=copy.deepcopy(root["targets"]),
        targets_sha=root["targets_sha"],
        reference_mse=root["reference_mse"],
        weights_sha=training.weight_sha(state["weights"]),
        optimizer_sha=identity(engine.state_identity(state["optimizer"])),
        balance_sha=identity(engine.state_identity(state["balance"])),
        head_sha=training.weight_sha({"head": state["head"]})
        if supervised(arm)
        else None,
        added_directions=C if arm in ["targeted", "random"] else 0,
        padded_rank=6,
        effective_concept_rank=6 if arm in ["targeted", "random"] else 5,
        effective_reserve_rank=5,
        allocated_core_parameters=sum(v.numel() for v in state["weights"].values()),
        usable_core_parameters=(G * 5 + (C if arm in ["targeted", "random"] else 0))
        * (2 * d + 1),
        total_training_updates=13640 + UPDATES,
        new_gradient_updates=UPDATES,
        final_adaptive_weights=None
        if not supervised(arm)
        else state["balance"]["log_weights"].exp().tolist(),
        phase_batch_sha=state["sampled_batches_sha"],
        phase_noise_sha=state["sampled_noise_sha"],
        reports=scored,
    )


def verify_endpoint(ctx, row):
    st = memory.required(ctx["store"], row["key"])
    capacity.matched.complete(st, UPDATES)
    capacity.matched.valid_weights(st["weights"], "vanilla", fit_context(ctx)["base"])
    root = capacity.source_row(ctx, row["seed"], supervised(row["arm"]))
    if (
        training.weight_sha(st["weights"]) != row["weights_sha"]
        or identity(row["targets"]) != row["targets_sha"]
        or identity(engine.state_identity(st["optimizer"])) != row["optimizer_sha"]
        or identity(engine.state_identity(st["balance"])) != row["balance_sha"]
        or root["weights_sha"] != row["starting_weights_sha"]
        or root["targets_sha"] != row["targets_sha"]
        or root["optimizer_sha"] != row["starting_optimizer_sha"]
        or root["balance_sha"] != row["starting_balance_sha"]
        or st["sampled_batches_sha"] != row["phase_batch_sha"]
        or st["sampled_noise_sha"] != row["phase_noise_sha"]
    ):
        raise ValueError("Reserve endpoint identity changed")
    assert_padding(
        st["weights"],
        ctx["base"]["groups"],
        len(row["targets"]),
        row["arm"] in ["targeted", "random"],
    )
    if supervised(row["arm"]):
        w = st["balance"]["log_weights"].exp()
        if training.weight_sha({"head": st["head"]}) != row["head_sha"] or not (
            0.0125 - 1e-7 <= w[0] <= 0.4 + 1e-7 and 0.0025 - 1e-7 <= w[1] <= 0.4 + 1e-7
        ):
            raise ValueError("Reserve head/controller identity differs")
    elif st["balance"] is not None:
        raise ValueError("Vanilla controller must be absent")
    del st
    gc.collect()


def preflight(ctx):
    old = ctx["store"].load("reserve_preflight")
    if old is not None:
        return old
    ts = torch.get_rng_state()
    cs = torch.cuda.get_rng_state_all()
    timings = {}
    try:
        for arm in ARMS:
            root, start = branch_start(ctx, arm, 21)
            sup = supervised(arm)
            sub = fit_context(ctx)
            sub["store"] = capacity.ScratchStore()
            torch.cuda.reset_peak_memory_stats()
            out = engine.fit(
                sub,
                "disposable_reserve_" + arm,
                start["weights"],
                root["targets"],
                10,
                111021,
                sup,
                sup,
                root["reference_mse"],
                initial_head=start["head"] if sup else None,
                initial_balance=start["balance"] if sup else None,
                initial_optimizer=start["optimizer"],
            )
            assert_padding(
                out["weights"],
                ctx["base"]["groups"],
                len(root["targets"]),
                arm in ["targeted", "random"],
            )
            timings[arm] = dict(
                seconds_per_update=out["seconds"] / 10,
                peak_allocated_GiB=torch.cuda.max_memory_allocated() / 2**30,
            )
            del start, out, sub
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        torch.set_rng_state(ts)
        torch.cuda.set_rng_state_all(cs)
    old = dict(
        timings=timings,
        gradient_only_hours=sum(
            v["seconds_per_update"] * UPDATES * len(SEEDS) for v in timings.values()
        )
        / 3600,
        note="Excludes TRAIN profiling, development scoring, checkpoint I/O and the later capacity sweep.",
    )
    ctx["store"].save("reserve_preflight", old)
    ctx["store"].write("preflight.json", old)
    return old


def write_tables(ctx, rows, phase):
    for split in ["select", "confirm"]:
        values = []
        for row in rows:
            primary = row["reports"][split]["primary"]
            diag = row["reports"][split]["fixed_category_slot_diagnostic"]
            values.append(
                dict(
                    seed=row["seed"],
                    arm=row["arm"],
                    phase=phase,
                    capture_pct=100 * primary["macro_capture"],
                    patch_AP_pct=100 * primary["macro_ap"],
                    VE_pct=100 * primary["reconstruction"]["ve"],
                    fixed_slot_capture_pct=100 * diag["macro_capture"],
                    fixed_slot_AP_pct=100 * diag["macro_ap"],
                )
            )
        if values:
            capacity.matched.csv_write(
                ctx["store"].results / f"{phase}_{split}_comparison.csv", values
            )
        means = []
        for arm in ARMS:
            rr = [v for v in values if v["arm"] == arm]
            if len(rr) == 3:
                means.append(
                    dict(
                        arm=arm,
                        **{
                            k: float(np.mean([v[k] for v in rr]))
                            for k in [
                                "capture_pct",
                                "patch_AP_pct",
                                "VE_pct",
                                "fixed_slot_capture_pct",
                                "fixed_slot_AP_pct",
                            ]
                        },
                    )
                )
        if means:
            capacity.matched.csv_write(
                ctx["store"].results / f"{phase}_{split}_means.csv", means
            )
        print(
            "Reserve " + phase + " " + split + " means / DEVELOPMENT ONLY:",
            json.dumps(means),
            flush=True,
        )


def compare(rows):
    output = []
    for seed in SEEDS:
        lookup = {r["arm"]: r for r in rows if r["seed"] == seed}
        for split in ["select", "confirm"]:
            for control in ["random", "ordinary"]:
                a = lookup["targeted"]["reports"][split]["primary"]
                b = lookup[control]["reports"][split]["primary"]
                output.append(
                    dict(
                        seed=seed,
                        split=split,
                        treatment="targeted",
                        control=control,
                        capture_gain_pp=100 * (a["macro_capture"] - b["macro_capture"]),
                        AP_gain_pp=100 * (a["macro_ap"] - b["macro_ap"]),
                        VE_gain_pp=100
                        * (a["reconstruction"]["ve"] - b["reconstruction"]["ve"]),
                        all_three_improve=bool(
                            a["macro_capture"] > b["macro_capture"]
                            and a["macro_ap"] > b["macro_ap"]
                            and a["reconstruction"]["ve"] > b["reconstruction"]["ve"]
                        ),
                        selection_native_eligible=bool(
                            lookup["targeted"]["reports"]["select"]["primary"][
                                "reconstruction"
                            ]["mse"]
                            <= lookup["targeted"]["reference_mse"] * 1.02
                        ),
                    )
                )
    return output


def run(ctx, source_zip=None):
    if "test" in ctx["packs"]:
        raise ValueError("Reserve screen is development only")
    old = ctx["store"].load("reserve_complete")
    if old is not None:
        if not old["complete"] or len(old["models"]) != 12:
            raise ValueError("Invalid completed reserve catalog")
        for row in old["models"]:
            verify_endpoint(ctx, row)
        return old
    for seed in SEEDS:
        selection_for(ctx, seed)
    if ctx["device"].startswith("cuda"):
        preflight(ctx)
    initial = []
    rows = []
    for arm in ARMS:
        for seed in SEEDS:
            root, start = branch_start(ctx, arm, seed)
            initial.append(initial_report(ctx, arm, seed, root, start))
            write_tables(ctx, initial, "initial")
            del start
            gc.collect()
            root, out = fit_branch(ctx, arm, seed)
            row = endpoint(ctx, arm, seed, root, out)
            verify_endpoint(ctx, row)
            rows.append(row)
            ctx["store"].save("reserve_progress", dict(complete=False, models=rows))
            ctx["store"].write(f"endpoints/{arm}_s{seed}.json", row)
            write_tables(ctx, rows, "final")
            del out
            gc.collect()
            torch.cuda.empty_cache()
        if source_zip is not None:
            export(ctx, source_zip)
    for seed in SEEDS:
        rr = [r for r in rows if r["seed"] == seed]
        for field in ["phase_batch_sha", "phase_noise_sha"]:
            if len({r[field] for r in rr}) != 1:
                raise ValueError("Reserve screen image/noise pairing differs")
    differences = compare(rows)
    capacity.matched.csv_write(
        ctx["store"].results / "development_gains.csv", differences
    )
    result = dict(
        complete=True,
        models=rows,
        comparisons=differences,
        initial_models=initial,
        recipe=settings(),
        development_only=True,
        test_features_opened=False,
        later_capacity_roots_unchanged=True,
    )
    ctx["store"].save("reserve_complete", result)
    ctx["store"].write("summary.json", result)
    capacity.matched.assert_sources(ctx)
    print(
        "RESERVE-ADDITION SCREEN COMPLETE. Full development outcomes saved; the independent capacity sweep follows.",
        flush=True,
    )
    if source_zip is not None:
        export(ctx, source_zip)
    return result


def export(ctx, source_zip):
    result = ctx["store"].load("reserve_complete")
    progress = ctx["store"].load("reserve_progress")
    ctx["store"].write(
        "summary.json",
        result
        if result is not None
        else dict(
            complete=False, recipe=settings(), progress=progress, development_only=True
        ),
    )
    path = capacity.matched.export(ctx, source_zip)
    target = path.with_name(
        "agf_reserve_additions_" + ctx["store"].run_id[:12] + "_results.zip"
    )
    path.replace(target)
    return target


def failure(ctx, stage, exc, source_zip):
    try:
        capacity.flush_active_checkpoint(ctx)
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            ),
        )
        print("Partial reserve evidence:", export(ctx, source_zip), flush=True)
    except BaseException as err:
        print("Reserve rescue/export failed:", type(err).__name__, str(err), flush=True)
