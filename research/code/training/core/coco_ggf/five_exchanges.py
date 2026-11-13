"""One bounded, resumable dependency/exchange pass on final ordinary 5D AGF-G.

Sources stay read-only. No optimizer, backbone, extraction, warm-up, readout fit,
or gradient update is called. Test features are opened after all three freezes.
"""

import copy, csv, gc, json, time, traceback, zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import core, inputs
from coco_baselines import training
from coco_baselines.storage import identity, sha_file
from coco_coordinated import run as memory
from . import five_continuation as five, cap_followup as cache, exchange5_ops as ops
from . import accumulating_regroup as guards, owner_routing as coverage, study

matched = five.matched
SEEDS = [21, 22, 23]
PARENT_RUN = "371d2d1d81ea6c9ce15fa17a6cdecc716f42304e385ac0cf9d2f3280572d6a0f"
PARENT_SOURCE_SHA = "4e2f00da8b218512d5a0d06241176819e14f884eeca1ac3269977a057594858b"


def settings():
    return dict(
        version=1,
        study="agf_final_five_direction_exchanges",
        seeds=SEEDS,
        groups=1024,
        block_rank=5,
        active_groups=16,
        screening_images=512,
        training_seed=107011,
        shard_images=25,
        top_incoming=5,
        candidate_pool=64,
        per_category_pool=2,
        screen_per_policy=8,
        max_attempts=8,
        max_accepted=5,
        positive_epsilon=1e-10,
        max_ap_drop=0.001,
        max_ve_drop=0.001,
        max_mse_growth=0.02,
        max_category_drop=0.01,
        proposal_score_scale=0.005,
        bootstrap=200,
        gradient_updates=0,
        new_warmups=0,
        readout_reassignment=False,
        original_native_limit="Original seed native development MSE x 1.02, checked on selection split only.",
        acceptance="Positive target and mean capture on TRAIN; positive mean capture and AP/VE/MSE/per-category preservation versus current and root on both historical development sets; original native-MSE limit on selection.",
        ordering="TRAIN-only dependency shortlist of eight, then exact standalone TRAIN gain order; no development/test reranking.",
        test_previously_inspected=True,
        scope="One bounded pass per seed; no claim of optimal search or independent generalization.",
    )


class SeedStore:
    def __init__(self, parent, seed):
        self.parent = parent
        self.prefix = "s" + str(seed) + "_"
        parent.check_mount()
        self.results = parent.results / ("seed" + str(seed))
        self.results.mkdir(exist_ok=True)

    def load(self, key):
        return self.parent.load(self.prefix + key)

    def save(self, key, value):
        return self.parent.save(self.prefix + key, value)

    def write(self, name, value):
        return self.parent.write("seed" + self.prefix[1:-1] + "/" + name, value)

    def event(self, stage, **kw):
        return self.parent.event(stage, seed=int(self.prefix[1:-1]), **kw)


def validate_parent_catalog(frozen, result):
    expected = {(seed, arm) for seed in SEEDS for arm in five.ARMS}
    if (
        not frozen.get("complete")
        or len(frozen.get("models", [])) != 9
        or {(r["seed"], r["arm"]) for r in frozen["models"]} != expected
    ):
        raise ValueError(
            "Require the complete frozen nine-model 5D continuation source"
        )
    if (
        not result.get("complete")
        or result.get("stage") != "continuation5"
        or len(result.get("models", [])) != 30
    ):
        raise ValueError("Require the completed all-seed 5D continuation audits")
    if (
        result.get("controls_retrained")
        or result.get("new_test_gradient_updates") != 0
        or result.get("new_structural_exchanges") != 0
    ):
        raise ValueError("Unexpected source protocol")
    for row in frozen["models"]:
        if (
            row["rank"] != 5
            or row["core_parameters"] != 7869440
            or row["maximum_active_coordinates"] != 80
            or row["total_training_updates"] != 13640
        ):
            raise ValueError("Wrong width, capacity, activity, or starting budget")
        if row["arm"] == "guided5_continue" and (
            not row["balanced"]
            or not row["supervised"]
            or row["pcgrad"]
            or row["decoder_pcgrad"]
        ):
            raise ValueError(
                "Exchanges must start from ordinary guided continuation, not PCGrad"
            )


def compact_matches(pack, ids, categories):
    if pack["image_ids"] != ids or pack["category_indices"] != categories:
        raise ValueError("Saved paired control cohort changed")
    if np.asarray(pack["error"]).shape != (len(ids), len(categories)):
        raise ValueError("Saved paired control dimensions changed")


def snapshot_from_state(row, state):
    return dict(
        spec=dict(
            kind="native",
            groups=1024,
            rank=5,
            active=16,
            targets=copy.deepcopy(row["targets"]),
            atom_ids=list(range(5120)),
        ),
        weights={k: v.detach().cpu().clone() for k, v in state["weights"].items()},
        head=state["head"].detach().cpu().clone(),
        source_weights_sha=row["weights_sha"],
        source_head_sha=row["head_sha"],
        source_balance_sha=row["balance_sha"],
        source_optimizer_sha=row["optimizer_sha"],
        reference_mse=row["reference_mse"],
    )


def assert_permutation(root, edited):
    a, b = root["spec"], edited["spec"]
    n = a["groups"] * a["rank"]
    order = b["atom_ids"]
    if sorted(order) != list(range(n)) or any(
        a[k] != b[k] for k in ["groups", "rank", "active", "targets"]
    ):
        raise ValueError(
            "An exchange changed ownership count, capacity, gate, or readouts"
        )
    ix = torch.tensor(order, dtype=torch.long)
    for key, expected in [
        ("W_enc", root["weights"]["W_enc"][:, ix]),
        ("W_dec", root["weights"]["W_dec"][ix]),
        ("b_enc", root["weights"]["b_enc"][ix]),
    ]:
        if not torch.equal(edited["weights"][key], expected):
            raise ValueError("Learned parameter values changed: " + key)
    if not torch.equal(root["head"], edited["head"]):
        raise ValueError("Trained classifier heads changed")
    for key in [
        "source_weights_sha",
        "source_head_sha",
        "source_balance_sha",
        "source_optimizer_sha",
        "reference_mse",
    ]:
        if root[key] != edited[key]:
            raise ValueError("Source optimizer/controller identity changed")


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
        raise ValueError("Final 5D source or numerical versions differ")
    frozen = memory.required(parent, "frozen_continuation5")
    result = memory.required(parent, "result_continuation5")
    validate_parent_catalog(frozen, result)
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
        raise ValueError("Unexpected original activation lineage")
    cfg = copy.deepcopy(parent.config["fit_config"])
    base = dict(parent.config["base"], block_rank=5)
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
        raise ValueError("Unexpected test features, normalization, or cohort")
    splits = [set(provenance["splits"][k]) for k in ["train", "select", "confirm"]] + [
        set(ids)
    ]
    if any(splits[i] & splits[j] for i in range(len(splits)) for j in range(i)):
        raise ValueError("Training/development/test overlap")
    if [len(provenance["splits"][k]) for k in ["train", "select", "confirm"]] != [
        2000,
        400,
        400,
    ]:
        raise ValueError("Unexpected cached split counts")
    root_rows = [r for r in frozen["models"] if r["arm"] == "guided5_continue"]
    if [r["seed"] for r in root_rows] != SEEDS:
        raise ValueError("Unexpected seed order")
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
        starting_models=[
            dict(
                seed=r["seed"],
                weights_sha=r["weights_sha"],
                targets_sha=r["targets_sha"],
            )
            for r in root_rows
        ],
        new_gradient_updates=0,
        controller="Saved cap-0.40 controller retained; no weight updates during Exchanges.",
    )
    store = inputs.JournalStore(output, payload, local / "rescue", require_drive)
    cache.install_drive_recovery(store)
    ctx = dict(
        store=store,
        sources=dict(
            continued5=parent, width_balance=width, original=original, strong=strong
        ),
        roots=root_rows,
        parent_catalog=frozen["models"],
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
        protocol=settings(),
    )
    for source, keys in [
        (parent, ["frozen_continuation5", "result_continuation5"]),
        (original, ["normalization"]),
        (strong, ["fresh_plan", "activation_identity_test"]),
    ]:
        ctx["checkpoint_stamps"] += [matched.source_key_stamp(source, k) for k in keys]
    for path in [original.results / "provenance.json"]:
        ctx["stamps"].append(dict(path=str(path), sha256=sha_file(path)))
    memory.memory_safe_metrics()
    for split in ["train", "select", "confirm"]:
        ctx["packs"][split] = cache.cached_split(ctx, split, cache_local)
    store.write("protocol.json", payload)
    store.write(
        "split_roles.json",
        dict(
            train=provenance["splits"]["train"],
            selection=provenance["splits"]["select"],
            confirmation=provenance["splits"]["confirm"],
            test=ids,
            test_features_opened=False,
            notes="Both development sets are reused acceptance sets. Test is previously inspected; no independent-confirmation claim.",
        ),
    )
    for row in root_rows:
        state = memory.required(parent, row["key"])
        matched.complete(state, 3000)
        five.check_state(ctx, row, state)
        matched.check_targets(row["targets"], cfg)
        snap = snapshot_from_state(row, state)
        assert_permutation(snap, snap)
        if training.weight_sha(snap["weights"]) != row["weights_sha"]:
            raise ValueError("Checkpoint conversion changed predictions")
        seedstore = SeedStore(store, row["seed"])
        old = seedstore.load("start")
        if old is None:
            seedstore.save("start", snap)
        else:
            assert_permutation(snap, old)
            if training.weight_sha(old["weights"]) != row["weights_sha"] or old["spec"][
                "atom_ids"
            ] != list(range(5120)):
                raise ValueError(
                    "Saved starting checkpoint is not the exact unedited source"
                )
        ctx["checkpoint_stamps"].append(matched.source_key_stamp(parent, row["key"]))
        del state, snap, old
        gc.collect()
    store.write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    matched.assert_sources(ctx)
    return ctx


@torch.no_grad()
def preflight(arm):
    root = arm["initial"]
    m = ops.model(arm, root)
    x = torch.as_tensor(
        np.array(arm["packs"]["train"]["x"][0], copy=True), device=arm["device"]
    )
    changed = ops.permute_snapshot(root, 0, root["spec"]["rank"])
    assert_permutation(root, changed)
    back = ops.permute_snapshot(changed, 0, root["spec"]["rank"])
    assert_permutation(root, back)
    if training.weight_sha(root["weights"]) != training.weight_sha(back["weights"]):
        raise ValueError("Swap is not reversible")
    edited = ops.model(arm, changed)
    raw = x @ m.W_enc + m.b_enc
    raw2 = x @ edited.W_enc + edited.b_enc
    relative = float(
        ((raw @ m.W_dec) - (raw2 @ edited.W_dec)).norm()
        / (raw @ m.W_dec).norm().clamp_min(1e-8)
    )
    if relative > 1e-5:
        raise ValueError("Ungated reconstruction changed under a pure permutation")
    arm["store"].write(
        "preflight.json",
        dict(
            groups=m.n_groups,
            rank=m.group_size,
            active_groups=m.l0,
            active_slots=m.l0 * m.group_size,
            ungated_relative_error=relative,
            swap_twice_restores=True,
            new_gradient_updates=0,
        ),
    )
    del m, edited
    gc.collect()


def measured(arm, key, snap, split):
    sig = identity(
        dict(
            weights=training.weight_sha(snap["weights"]),
            spec=snap["spec"],
            ids=arm["packs"][split]["ids"],
            cfg=arm["cfg"],
        )
    )
    s = arm["store"]
    old = s.load(key)
    if old is not None:
        if old["signature"] != sig:
            raise ValueError("Resumed score identity changed: " + key)
        return old["report"]
    start = time.perf_counter()
    m = ops.model(arm, snap)
    s.event("scoring", key=key, split=split, images=len(arm["packs"][split]["ids"]))
    with torch.inference_mode():
        report = core.validation(
            m, arm["packs"][split], snap["spec"]["targets"], arm["cfg"], arm["device"]
        )
    cache.bound.validate_metrics(report)
    s.save(key, dict(signature=sig, report=report))
    s.event(
        "scored",
        key=key,
        seconds=time.perf_counter() - start,
        capture_pct=100 * report["macro_capture"],
        AP_pct=100 * report["macro_ap"],
        VE_pct=100 * report["reconstruction"]["ve"],
    )
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return report


def native_gate(report, current, root, rule, reference_mse=None, target=None):
    result = guards.positive_gate(report, current, root, rule, target)
    if reference_mse is not None:
        margin = reference_mse * 1.02 - report["reconstruction"]["mse"]
        result["margins"]["original_native_MSE"] = margin
        if not np.isfinite(margin) or margin < -1e-12:
            result["failures"].append("original_native_MSE")
    result["passed"] = not result["failures"]
    return result


@torch.no_grad()
def direction_profiles(arm):
    s = arm["store"]
    old = s.load("profiles_train_screen")
    if old is not None:
        return old
    m = ops.model(arm, arm["initial"])
    p = arm["packs"]["train_screen"]
    rule = arm["protocol"]
    total = ops.empty_stats(len(m.targets), m.total_atoms, m.group_size)
    for lo in range(0, len(p["ids"]), rule["shard_images"]):
        hi = min(lo + rule["shard_images"], len(p["ids"]))
        key = f"profiles_{lo:05d}"
        shard = s.load(key)
        if shard is None:
            shard = ops.shard_stats(m, p, lo, hi, arm["cfg"])
            shard["image_ids"] = p["ids"][lo:hi]
            s.save(key, shard)
        if shard["image_ids"] != p["ids"][lo:hi]:
            raise ValueError("Profile shard image order changed")
        for k in total:
            total[k] += shard[k]
        s.event("direction_profiles", done=hi, total=len(p["ids"]))
    if (total["counts"] == 0).any() or (total["energy"] <= 0).any():
        raise ValueError("Missing declared training category evidence")
    for key in ["gain", "remove", "cross"]:
        total[key] /= total["energy"].reshape((-1,) + (1,) * (total[key].ndim - 1))
    for key in ["raw_fg", "raw_bg", "gated_fg", "signed_delta", "exclusion"]:
        total[key] /= total["counts"][:, None]
    total["category_indices"] = np.array([t["category_index"] for t in m.targets])
    total["readout_groups"] = np.array([t["contrast_group"] for t in m.targets])
    s.save("profiles_train_screen", total)
    del m
    gc.collect()
    return total


def select_arm(arm):
    s = arm["store"]
    root = arm["initial"]
    old = s.load("selection_frozen")
    if old is not None:
        if old["root_sha"] != training.weight_sha(root["weights"]):
            raise ValueError("Resumed root changed")
        final = memory.required(s, "final_agfx")
        assert_permutation(root, final)
        if training.weight_sha(final["weights"]) != old["final_sha"]:
            raise ValueError("Resumed selected model changed")
        return old
    if "test" in arm["packs"]:
        raise ValueError("Test features must stay closed throughout selection")
    rule = arm["protocol"]
    targets = root["spec"]["targets"]
    train = arm["packs"]["train"]
    allocation = dict(
        coverage.settings(),
        assignment_images=rule["screening_images"],
        training_seed=rule["training_seed"],
    )
    ix, evidence = coverage.assignment_indices(
        train, [t["category_index"] for t in targets], allocation, arm["cfg"]
    )
    screen = dict(
        x=np.array(train["x"][ix], copy=True),
        ids=[train["ids"][i] for i in ix],
        masks=[train["masks"][i] for i in ix],
    )
    arm["packs"] = dict(arm["packs"], train_screen=screen)
    s.write("training_screen_coverage.json", evidence)
    preflight(arm)
    refs = {
        split: measured(arm, split + "_root", root, split)
        for split in ["train_screen", "select", "confirm"]
    }
    source_dev = arm["source_row"]["development"]
    for key in ["mse", "ve"]:
        if (
            abs(
                refs["select"]["reconstruction"][key]
                - source_dev["reconstruction"][key]
            )
            > 1e-5
        ):
            raise ValueError("Starting development reconstruction does not reproduce")
    if refs["select"]["reconstruction"]["mse"] > root["reference_mse"] * 1.02 + 1e-12:
        raise ValueError(
            "The selected ordinary 5D starting checkpoint fails its original native limit"
        )
    pool = ops.candidate_pool(direction_profiles(arm), targets, rule)
    order = s.load("proposal_order")
    graph = None
    if order is None:
        screened = []
        if pool:
            graph = ops.build_graph(arm, pool, screen)
            ranked = ops.graph_rank(pool, graph, rule)[: rule["screen_per_policy"]]
            s.write("graph_ranked_shortlist.json", ranked)
            for p in ranked:
                candidate = ops.permute_snapshot(root, p["outgoing"], p["incoming"])
                report = measured(
                    arm, "standalone_" + p["id"], candidate, "train_screen"
                )
                target = p["category_index"]
                a = {r["category_index"]: r for r in report["categories"]}
                b = {r["category_index"]: r for r in refs["train_screen"]["categories"]}
                screened.append(
                    dict(
                        proposal=p,
                        mean_gain=report["macro_capture"]
                        - refs["train_screen"]["macro_capture"],
                        target_gain=a[target]["capture"] - b[target]["capture"],
                    )
                )
            screened.sort(
                key=lambda r: (-r["mean_gain"], -r["target_gain"], r["proposal"]["id"])
            )
        order = [r["proposal"] for r in screened[: rule["max_attempts"]]]
        s.save("proposal_order", order)
        s.write("standalone_training_screen.json", screened)
        s.write("proposal_order.json", order)
    if order and graph is None:
        graph = memory.required(s, "dependency_graph")
    current = root
    current_reports = dict(refs)
    steps = []
    accepted = []
    groups = set()
    used_targets = set()
    for i, p in enumerate(order):
        key = f"exchange{i:02d}"
        before = training.weight_sha(current["weights"])
        row = s.load(key)
        if row is not None:
            if row["before_sha"] != before or identity(row["proposal"]) != identity(p):
                raise ValueError("Resumed decision chain changed")
        else:
            skip = guards.conflict(p, groups, used_targets)
            if len(accepted) >= rule["max_accepted"]:
                skip = "maximum_accepted_reached"
            row = dict(
                proposal=p, before_sha=before, accepted=False, skip=skip, checks={}
            )
            if skip is None:
                candidate = ops.permute_snapshot(current, p["outgoing"], p["incoming"])
                assert_permutation(root, candidate)
                tr = measured(arm, key + "_train", candidate, "train_screen")
                tg = native_gate(
                    tr,
                    current_reports["train_screen"],
                    refs["train_screen"],
                    rule,
                    target=p["category_index"],
                )
                row.update(
                    train_report=tr,
                    train_gate=tg,
                    candidate_sha=training.weight_sha(candidate["weights"]),
                )
                if tg["passed"]:
                    for split in ["select", "confirm"]:
                        report = measured(arm, key + "_" + split, candidate, split)
                        gate = native_gate(
                            report,
                            current_reports[split],
                            refs[split],
                            rule,
                            root["reference_mse"] if split == "select" else None,
                        )
                        row["checks"][split] = dict(report=report, gate=gate)
                    row["accepted"] = all(
                        x["gate"]["passed"] for x in row["checks"].values()
                    )
                    row["skip"] = (
                        None
                        if row["accepted"]
                        else "development_gain_or_preservation_failed"
                    )
                else:
                    row["skip"] = "training_gain_or_preservation_failed"
                if row["accepted"]:
                    m = ops.model(arm, candidate)
                    row["gate_effects"] = ops.gate_effects(arm, m, screen, graph, p)
                    del m
                    s.save(key + "_accepted", candidate)
            row["after_sha"] = row.get("candidate_sha") if row["accepted"] else before
            s.save(key, row)
        if row["accepted"]:
            current = memory.required(s, key + "_accepted")
            assert_permutation(root, current)
            if training.weight_sha(current["weights"]) != row["after_sha"]:
                raise ValueError("Accepted model changed")
            current_reports = dict(
                train_screen=row["train_report"],
                **{k: v["report"] for k, v in row["checks"].items()},
            )
            accepted.append(p)
            groups.update((p["receiver_group"], p["donor_group"]))
            used_targets.add(p["category_index"])
        elif training.weight_sha(current["weights"]) != before:
            raise ValueError("Rejection mutated current model")
        steps.append(row)
        s.write("steps/" + key + ".json", row)
        s.event(
            "exchange_decision",
            done=i + 1,
            total=len(order),
            accepted=row["accepted"],
            accepted_total=len(accepted),
            reason=row["skip"],
        )
    assert_permutation(root, current)
    s.save("final_agfx", current)
    out = dict(
        seed=arm["seed"],
        root_sha=training.weight_sha(root["weights"]),
        final_sha=training.weight_sha(current["weights"]),
        accepted_proposals=accepted,
        attempted=len(order),
        steps=steps,
        development={k: v for k, v in current_reports.items() if k != "train_screen"},
        original_native_eligible=current_reports["select"]["reconstruction"]["mse"]
        <= root["reference_mse"] * 1.02,
        targets_sha=identity(root["spec"]["targets"]),
        head_sha=root["source_head_sha"],
        new_gradient_updates=0,
        groups=root["spec"]["groups"],
        rank=root["spec"]["rank"],
        active_groups=root["spec"]["active"],
        maximum_active_coordinates=root["spec"]["rank"] * root["spec"]["active"],
        total_training_updates=13640,
    )
    s.save("selection_frozen", out)
    s.write("selection_frozen.json", out)
    arm["packs"].pop("train_screen")
    del screen
    gc.collect()
    return out


def run(ctx, source_path=None):
    if "test" in ctx["packs"]:
        raise ValueError("Close test features before selection")
    rows = []
    for row in ctx["roots"]:
        seed = row["seed"]
        s = SeedStore(ctx["store"], seed)
        arm = dict(
            ctx, store=s, seed=seed, initial=memory.required(s, "start"), source_row=row
        )
        rows.append(select_arm(arm))
        ctx["store"].event(
            "seed_frozen", seed=seed, accepted=len(rows[-1]["accepted_proposals"])
        )
        if source_path is not None:
            export(ctx, source_path)
        del arm
        gc.collect()
        torch.cuda.empty_cache()
    frozen = dict(
        complete=True, models=rows, test_features_opened=False, new_gradient_updates=0
    )
    old = ctx["store"].load("exchanges_frozen")
    if old is not None and identity(old) != identity(frozen):
        raise ValueError("Frozen final catalog changed")
    ctx["store"].save("exchanges_frozen", frozen)
    ctx["store"].write("exchanges_frozen.json", frozen)
    verify_freeze(ctx)
    matched.assert_sources(ctx)
    return frozen


def verify_freeze(ctx):
    frozen = memory.required(ctx["store"], "exchanges_frozen")
    if (
        not frozen["complete"]
        or len(frozen["models"]) != 3
        or [r["seed"] for r in frozen["models"]] != SEEDS
    ):
        raise ValueError("Freeze all three AGF-X endpoints before test access")
    for row in frozen["models"]:
        s = SeedStore(ctx["store"], row["seed"])
        root = memory.required(s, "start")
        edited = memory.required(s, "final_agfx")
        assert_permutation(root, edited)
        if (
            training.weight_sha(edited["weights"]) != row["final_sha"]
            or identity(edited["spec"]["targets"]) != row["targets_sha"]
        ):
            raise ValueError("Frozen model or readouts changed")
    return frozen


def load_test(ctx):
    verify_freeze(ctx)
    # Reuse the established memory-mapped shard loader; its legacy three-seed
    # barrier is satisfied by the actual starting catalog, not new cap fitting.
    sub = dict(ctx, cap040_rows=ctx["roots"])
    s = ctx["store"]

    class Barrier:
        def load(self, key):
            if key == "all_models_frozen":
                return dict(complete=True, models=ctx["roots"])
            return s.load(key)

        def event(self, *a, **kw):
            return s.event(*a, **kw)

    sub["store"] = Barrier()
    old_verify = cache.verify_test_freeze

    def verify(actual, frozen):
        if identity(frozen["models"]) != identity(ctx["roots"]):
            raise ValueError("Starting catalog changed")
        verify_freeze(ctx)

    cache.verify_test_freeze = verify
    try:
        return cache.load_test(sub)
    finally:
        cache.verify_test_freeze = old_verify


def saved_control(ctx, seed, arm):
    source = ctx["sources"]["continued5"]
    row = next(
        r for r in ctx["parent_catalog"] if r["seed"] == seed and r["arm"] == arm
    )
    tag = f"s{seed}_{arm}"
    a = memory.required(source, "audit_" + tag)
    compact = memory.required(source, "compact_" + tag)
    if a["identity"] != cache.report_signature(row, ctx["test_stamp"], ctx["cfg"]):
        raise ValueError("Saved control test signature changed")
    cache.check_report(a["report"], row, ctx["cfg"])
    compact_matches(compact, ctx["test_ids"], ctx["cfg"]["category_indices"])
    ctx["checkpoint_stamps"] += [
        matched.source_key_stamp(source, k) for k in ["audit_" + tag, "compact_" + tag]
    ]
    return a["report"], compact


def audit_edited(ctx, row, control, control_compact):
    s = ctx["store"]
    seed = row["seed"]
    key = f"audit_agfx5_s{seed}"
    sig = identity(
        dict(
            weights=row["final_sha"],
            targets=row["targets_sha"],
            test=ctx["test_stamp"],
            cfg=ctx["cfg"],
        )
    )
    old = s.load(key)
    if old is not None:
        if old["identity"] != sig:
            raise ValueError("Cached edited test audit changed")
        compact = (
            control_compact
            if row["final_sha"] == row["root_sha"]
            else memory.required(s, f"compact_agfx5_s{seed}")
        )
        return old["report"], compact
    if row["final_sha"] == row["root_sha"]:
        report = copy.deepcopy(control)
        compact = control_compact
    else:
        snap = memory.required(SeedStore(s, seed), "final_agfx")
        m = ops.model(ctx, snap)
        with torch.inference_mode(), matched.progress(ctx, m, f"agfx5_s{seed}"):
            report = core.validation(
                m,
                ctx["packs"]["test"],
                snap["spec"]["targets"],
                ctx["cfg"],
                ctx["device"],
                with_arrays=True,
            )
            compact = study.compact_audit(
                report.pop("_audit"),
                matched.mapping(snap["spec"]["targets"]),
                ctx["cfg"],
            )
        compact_matches(compact, ctx["test_ids"], ctx["cfg"]["category_indices"])
        s.save(f"compact_agfx5_s{seed}", compact)
        del m, snap
        gc.collect()
        torch.cuda.empty_cache()
    report.update(
        seed=seed,
        arm="agfx5",
        label="AGF-X 5D: Exchanges",
        rank=5,
        origin="final_ordinary_5D_guided_checkpoint",
        weights_sha=row["final_sha"],
        targets_sha=row["targets_sha"],
        core_parameters=7869440,
        head_parameters=150,
        maximum_active_coordinates=80,
        total_training_updates=13640,
        new_gradient_updates=0,
        accepted_exchanges=len(row["accepted_proposals"]),
        legacy_native_eligible=row["original_native_eligible"],
        development_reconstruction=row["development"]["select"]["reconstruction"],
        common_native_MSE_limit=next(r for r in ctx["roots"] if r["seed"] == seed)[
            "reference_mse"
        ]
        * 1.02,
    )
    cache.bound.validate_metrics(report)
    s.save(key, dict(identity=sig, report=report))
    s.write(f"reports/agfx5_s{seed}.json", report)
    return report, compact


def evaluate(ctx):
    frozen = verify_freeze(ctx)
    matched.assert_sources(ctx)
    old = ctx["store"].load("experiment_complete")
    if old is not None:
        return old
    load_test(ctx)
    rows = []
    comparisons = []
    for row in frozen["models"]:
        seed = row["seed"]
        control, compact = saved_control(ctx, seed, "guided5_continue")
        edited, edited_compact = audit_edited(ctx, row, control, compact)
        key = f"paired_agfx5_s{seed}"
        paired = ctx["store"].load(key)
        if paired is None:
            if row["final_sha"] == row["root_sha"]:
                paired = dict(
                    capture_gain_ci95=[0.0, 0.0],
                    ap_gain_ci95=[0.0, 0.0],
                    per_category=[
                        dict(
                            category_index=c,
                            capture_gain_ci95=[0.0, 0.0],
                            ap_gain_ci95=[0.0, 0.0],
                        )
                        for c in ctx["cfg"]["category_indices"]
                    ],
                    valid_bootstrap_replicates=0,
                    invalid_full_cohort_replicates=0,
                    note="Identical unchanged models: every defined image-resample difference is exactly zero; no bootstrap draws needed.",
                )
            else:
                ctx["store"].event("paired_bootstrap_start", seed=seed, draws=200)
                paired = study.paired(
                    edited_compact, compact, dict(ctx["cfg"], bootstrap=200)
                )
            ctx["store"].save(key, paired)
        test_gate = native_gate(edited, control, control, settings())
        comparisons.append(
            dict(
                seed=seed,
                accepted_exchanges=len(row["accepted_proposals"]),
                capture_gain_pp=100
                * (edited["macro_capture"] - control["macro_capture"]),
                AP_gain_pp=100 * (edited["macro_ap"] - control["macro_ap"]),
                VE_gain_pp=100
                * (edited["reconstruction"]["ve"] - control["reconstruction"]["ve"]),
                native_eligible=row["original_native_eligible"],
                test_checks=test_gate,
                paired_intervals=paired,
                test_used_for_selection=False,
            )
        )
        del compact, edited_compact
        gc.collect()
        vanilla, vc = saved_control(ctx, seed, "vanilla5_continue")
        del vc
        rows.extend([control, edited, vanilla])
        ctx["store"].write(f"reports/agfg5_s{seed}.json", control)
        ctx["store"].write(f"reports/vanilla5_s{seed}.json", vanilla)
        ctx["store"].write(f"paired/agfx5_s{seed}.json", comparisons[-1])
    out = dict(
        complete=True,
        models=rows,
        comparisons=comparisons,
        seeds=SEEDS,
        new_gradient_updates=0,
        new_backbone_extractions=0,
        controls_retrained=False,
        category_readouts_refitted=False,
        all_models_frozen_before_test=True,
        test_previously_inspected=True,
        structural_advantage_all_seeds=all(
            r["accepted_exchanges"] > 0
            and r["test_checks"]["passed"]
            and r["native_eligible"]
            for r in comparisons
        ),
        scope="Three fitted seeds on one inspected cohort. Image intervals condition on fitted models; no seed-level/selection/multiplicity adjustment. No image-probe or downstream-improvement claim.",
    )
    tables(ctx, out)
    matched.assert_sources(ctx)
    ctx["store"].save("experiment_complete", out)
    ctx["store"].write("summary.json", out)
    return out


def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)


def tables(ctx, out):
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
    folder = ctx["store"].results
    csv_write(folder / "comparison.csv", rows)
    means = []
    for arm, label in [
        ("guided5_continue", "AGF-G 5D: unchanged"),
        ("agfx5", "AGF-X 5D: Exchanges"),
        ("vanilla5_continue", "Vanilla BSF 5D: reused control"),
    ]:
        rr = [r for r in rows if r["arm"] == arm]
        if len(rr) != 3:
            raise ValueError("All three seeds must appear in means")
        means.append(
            dict(
                arm=arm,
                label=label,
                **{
                    k: float(np.mean([r[k] for r in rr]))
                    for k in ["capture_pct", "patch_AP_pct", "VE_pct"]
                },
                native_eligible_seeds=sum(r["native_eligible"] for r in rr),
            )
        )
    csv_write(folder / "means.csv", means)
    gains = []
    for r in out["comparisons"]:
        ci = r["paired_intervals"]
        gains.append(
            dict(
                seed=r["seed"],
                accepted_exchanges=r["accepted_exchanges"],
                **{k: r[k] for k in ["capture_gain_pp", "AP_gain_pp", "VE_gain_pp"]},
                capture_ci_low=100 * ci["capture_gain_ci95"][0],
                capture_ci_high=100 * ci["capture_gain_ci95"][1],
                AP_ci_low=100 * ci["ap_gain_ci95"][0],
                AP_ci_high=100 * ci["ap_gain_ci95"][1],
                test_checks_passed=r["test_checks"]["passed"],
            )
        )
    csv_write(folder / "paired.csv", gains)
    cats = [
        dict(seed=r["seed"], arm=r["arm"], **c)
        for r in out["models"]
        for c in r["categories"]
    ]
    # Category dictionaries have identical schema for these native models.
    csv_write(folder / "categories.csv", cats)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    colors = ["#7951A0", "#007D75", "#527E9D"]
    for ax, key, title in zip(
        axs,
        ["capture_pct", "patch_AP_pct", "VE_pct"],
        ["Capture", "Patch AP", "Reconstruction VE"],
    ):
        bars = ax.bar(range(3), [r[key] for r in means], color=colors)
        ax.bar_label(bars, fmt="%.3f%%", padding=3)
        ax.set(
            xticks=range(3),
            xticklabels=["AGF-G", "AGF-X", "Vanilla"],
            ylim=(0, 100),
            title=title,
            ylabel="Percent",
        )
    (folder / "figures").mkdir(exist_ok=True)
    fig.savefig(folder / "figures/means.png", dpi=180)
    fig.savefig(folder / "figures/means.pdf")
    plt.close(fig)
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), layout="constrained")
    for ax, key, interval, title in zip(
        axs,
        ["capture_gain_pp", "AP_gain_pp", "VE_gain_pp"],
        ["capture", "AP", None],
        ["Capture change", "Patch AP change", "VE change"],
    ):
        for i, r in enumerate(gains):
            ax.scatter(i, r[key], color="#007D75")
            if interval:
                ax.plot(
                    [i, i],
                    [r[interval + "_ci_low"], r[interval + "_ci_high"]],
                    color="#007D75",
                )
        ax.axhline(0, color="gray", lw=0.8)
        ax.set(
            xticks=range(3),
            xticklabels=SEEDS,
            xlabel="Seed",
            ylabel="Percentage points",
            title=title,
        )
    fig.savefig(folder / "figures/gains.png", dpi=180)
    fig.savefig(folder / "figures/gains.pdf")
    plt.close(fig)
    print("Three-seed test means:", json.dumps(means, indent=2), flush=True)


def export(ctx, source_path):
    # Atomic replacement prevents an interruption leaving the last rescue ZIP corrupt.
    s = ctx["store"]
    path = s.output / f"agf_5d_exchanges_{s.run_id[:12]}_results.zip"
    tmp = path.with_suffix(".pending.zip")
    completed = []
    for seed in SEEDS:
        saved = SeedStore(s, seed).load("selection_frozen")
        if saved is not None:
            completed.append(
                dict(
                    seed=seed,
                    accepted_exchanges=len(saved["accepted_proposals"]),
                    attempted=saved["attempted"],
                )
            )
    s.write(
        "progress_summary.json",
        dict(
            complete=s.load("experiment_complete") is not None,
            completed_selection=completed,
            planned_seeds=SEEDS,
            new_gradient_updates=0,
            all_models_frozen=s.load("exchanges_frozen") is not None,
        ),
    )
    s.write(
        "checkpoint_locations.json",
        {
            k: dict(run=v.run_id, checkpoints=str(v.root))
            for k, v in ctx["sources"].items()
        }
        | {"current": dict(run=s.run_id, checkpoints=str(s.root))},
    )
    hashes = {}
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(s.results.rglob("*")):
            if file.is_file():
                name = "results/" + file.relative_to(s.results).as_posix()
                z.write(file, name)
                hashes[name] = sha_file(file)
        z.write(source_path, "source.zip")
        hashes["source.zip"] = sha_file(source_path)
        z.writestr("artifact_hashes.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:
            raise OSError("Results ZIP CRC check failed")
        import hashlib

        if any(hashlib.sha256(z.read(n)).hexdigest() != h for n, h in hashes.items()):
            raise OSError("Results ZIP hash check failed")
    tmp.replace(path)
    return path


def failure(ctx, stage, exc, source_path):
    try:
        ctx["store"].write(
            "failure.json",
            dict(
                stage=stage,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
                resume="Rerun unchanged. Completed profile/graph shards, reports, and committed exchange decisions are reused.",
            ),
        )
        print("Partial evidence:", export(ctx, source_path), flush=True)
    except BaseException as err:
        print("Partial export failed:", type(err).__name__, str(err), flush=True)
