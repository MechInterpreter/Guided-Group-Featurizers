"""Remeasure dependencies at saved GGF u600; frozen AGF, no new training.

One training-chosen batch, one development check, one historical confirmation.
Post-batch graph is diagnostic only and cannot change the frozen nominee.
"""

import copy
import gc
import json
import time
from pathlib import Path
import numpy as np
from coco_adaptation import inputs
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import accumulating_regroup as accum, dependency_regroup as dep
from . import direction_regroup as prior, bank_experiment as bank, owner_routing as old

PARENT_RUN = "bfc7808602f266a79120712f17f6761a604538f529c65c8aff1e031469aa06df"
PARENT_SHA = "7fc600132e0e91b63019e56f4863828d301e7f25310b7b66fbf90581951204c6"
START_KEY = "search_budget_continued_ggf_u600"


def settings():
    cfg = accum.settings(False)
    cfg.update(
        study="mature_ggf_frozen_dependency_remeasurement",
        version=1,
        starting_checkpoint=START_KEY,
        common_start_updates=600,
        proposal_score_scale=0.005,
        post_batch_analysis=True,
        scope="One seed at saved u600; zero new gradient updates. Historical development and disjoint historical confirmation; final test closed.",
    )
    return cfg


def validate_start(source, initial):
    """Fail closed on an absent, incomplete or different stronger parent."""
    complete = memory.required(source, "experiment_complete")
    state = memory.required(source, "search_budget_continued_ggf")
    snap = memory.required(source, START_KEY)
    if not complete["complete"] or state["updates"] != 600:
        raise ValueError(
            "The saved 600-update GGF is not complete; no retraining fallback"
        )
    if accum.weight_sha(state["current"]) != accum.weight_sha(snap):
        raise ValueError("u600 snapshot differs from the completed continuation state")
    if identity(snap["spec"]) != identity(initial["spec"]):
        raise ValueError(
            "Stronger checkpoint changed group/readout/capacity definitions"
        )
    history = complete["budget_control_history"]
    if not any(r["checkpoint"] == START_KEY and r["update"] == 600 for r in history):
        raise ValueError("Saved parent history does not contain u600")
    return snap


def prepare(
    parent,
    accumulation_parent,
    output,
    local,
    sha,
    versions,
    device,
    require_drive=True,
):
    cfg = settings()
    output = Path(output)
    local = Path(local)
    a = output.resolve()
    b = Path(accumulation_parent).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError(
            "New output must be separate from read-only accumulation parent"
        )
    ctx = old.prepare(
        parent,
        output / "inputs",
        local,
        sha,
        versions,
        device=device,
        require_drive=require_drive,
    )
    w, h, t, row = old.source_start(ctx, cfg["seed"])
    original = bank.native_snapshot(ctx, w, t, h, 1024, 3, 16)
    source = memory.FrozenStore(
        b / "checkpoints" / PARENT_RUN[:20], local / "read_accumulation"
    )
    if source.run_id != PARENT_RUN or source.config["source_sha"] != PARENT_SHA:
        raise ValueError("Wrong completed accumulation run/source")
    roles = json.loads((source.results / "split_roles.json").read_text())
    if (
        roles["source_weights_sha"] != row["weights_sha"]
        or roles["category_readouts"] != t
    ):
        raise ValueError("Parent does not descend from expected GGF weights/readouts")
    for name in ("train", "select"):
        if roles["ids"][name] != ctx["packs"][name]["ids"]:
            raise ValueError("Parent image identities changed")
    ctx["initial"] = validate_start(source, original)
    ctx["protocol"] = cfg
    ctx["prior"] = None
    ctx["cfg"] = dict(
        ctx["eval_cfg"],
        **{
            k: cfg[k]
            for k in (
                "lr",
                "snr",
                "lambda_capture",
                "lambda_detection",
                "grad_clip",
                "images_per_update",
            )
        },
    )
    ctx["source_weights_sha"] = accum.weight_sha(ctx["initial"])
    ctx["stamps"] += memory.source_snapshot([source])
    ctx["strong_parent"] = source
    ctx["store"] = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=sha,
            versions=versions,
            device=device,
            parent_run=ctx["source"].run_id,
            activation_run=ctx["original"].run_id,
            accumulation_run=PARENT_RUN,
            start_weights_sha=ctx["source_weights_sha"],
        ),
        local / "mature_rescue",
        require_drive,
    )
    s = ctx["store"]
    s.save("verified_start", ctx["initial"])
    s.write(
        "split_roles.json",
        dict(
            ids={k: p["ids"] for k, p in ctx["packs"].items()},
            source_checkpoint=START_KEY,
            source_weights_sha=ctx["source_weights_sha"],
            category_readouts=t,
            train="Fresh profiles/graph at u600, exact standalone training ranking, cumulative actual TopK checks; diagnostic post-batch graph.",
            select="400 reused development images; check one frozen batch, no prefix/ranking selection.",
            confirm="Historical disjoint confirmation only if frozen batch eligible; no second nominee.",
            final_test_opened=False,
        ),
    )
    s.event(
        "prepared",
        source_checkpoint=START_KEY,
        source_weights_sha=ctx["source_weights_sha"],
        historical_versions=source.config["versions"],
        current_versions=versions,
        reused_historical_profiles=False,
        new_gradient_updates=0,
    )
    return ctx


def screen_and_graph(ctx):
    """Same successful training-priority recipe, all statistics rebuilt at u600."""
    s = ctx["store"]
    done = s.load("fresh_screen")
    if done is not None:
        ctx["dependency"] = s
        ctx["saved_screen"] = done
        return done
    train = dep.profiles(ctx, "train")
    ranking_cfg = dict(
        ctx["protocol"], minimum_target_gain=ctx["protocol"]["proposal_score_scale"]
    )
    pool = dep.candidate_pool(train, ctx["initial"]["spec"]["targets"], ranking_cfg)
    s.write("candidate_pool.json", pool)
    screen = dep.training_screen(ctx)
    if not pool:
        done = dict(
            rows=[], empty_reason="No positive raw/additive training candidates"
        )
        s.save("fresh_screen", done)
        ctx["dependency"] = s
        ctx["saved_screen"] = done
        return done
    graph = dep.build_graph(ctx, pool, screen)
    ranked = dep.graph_rank(pool, graph, ranking_cfg)
    s.write("dependencies/ranked_exchanges.json", ranked)
    view = dict(ctx, packs=dict(ctx["packs"], train_screen=screen))
    base = accum.measured(view, "standalone_root", ctx["initial"], "train_screen")
    rows = []
    for p in ranked[: ctx["protocol"]["max_attempts"]]:
        snap = prior.permute_snapshot(ctx["initial"], p["outgoing"], p["incoming"])
        report = accum.measured(view, "standalone_" + p["id"], snap, "train_screen")
        category = p["category_index"]
        a = accum._categories(report)
        b = accum._categories(base)
        row = dict(
            proposal=p,
            report=report,
            in_graph_shortlist=True,
            mean_capture_delta=report["macro_capture"] - base["macro_capture"],
            target_capture_delta=a[category]["capture"] - b[category]["capture"],
            gate=accum.positive_gate(report, base, base, ctx["protocol"], category),
            all_category_deltas=accum.category_deltas(report, base),
        )
        rows.append(row)
        s.write("standalone/" + p["id"] + ".json", row)
        s.event(
            "standalone_screen",
            done=len(rows),
            total=min(len(ranked), ctx["protocol"]["max_attempts"]),
            proposal=p["id"],
            gain_pp=100 * row["mean_capture_delta"],
            safe=row["gate"]["passed"],
        )
    done = dict(
        rows=rows,
        selection_rule="New u600 training graph shortlist, then exact standalone training mean gain; actual cumulative gates decide adoption.",
        pool_size=len(pool),
        graph_directions=len(graph["atom_ids"]),
        weights_sha=ctx["source_weights_sha"],
    )
    s.save("fresh_screen", done)
    s.write("fresh_screen.json", done)
    ctx["dependency"] = s
    ctx["saved_screen"] = done
    del screen
    gc.collect()
    return done


def accumulate(ctx):
    if not ctx["saved_screen"]["rows"]:
        s = ctx["store"]
        s.save("accumulation_root", ctx["initial"])
        s.save("accumulated_snapshot", ctx["initial"])
        screen = dep.training_screen(ctx)
        view = dict(ctx, packs=dict(ctx["packs"], train_screen=screen))
        root = accum.measured(view, "train_root", ctx["initial"], "train_screen")
        out = dict(
            steps=[],
            accepted_proposals=[],
            gradient_updates=0,
            final_weight_sha=accum.weight_sha(ctx["initial"]),
            initial_report=root,
            final_report=root,
            final_test_opened=False,
        )
        s.save("accumulation_complete", out)
        s.write("accumulation.json", out)
        return out
    return accum.accumulate(ctx)


class NamespaceStore:
    """Separate resumable post-batch statistics from the original u600 graph."""

    def __init__(self, store):
        self.parent = store
        self.results = store.results / "post_analysis"
        self.results.mkdir(exist_ok=True)

    def load(self, key):
        return self.parent.load("post_" + key)

    def save(self, key, value):
        return self.parent.save("post_" + key, value)

    def write(self, path, value):
        return self.parent.write("post_analysis/" + path, value)

    def event(self, name, **kw):
        return self.parent.event("post_" + name, **kw)


def reanalyze(ctx):
    """Fresh graph at the final TRAINING-adopted layout; no new edit is applied."""
    s = ctx["store"]
    cached = s.load("post_batch_analysis")
    if cached is not None:
        return cached
    snap = memory.required(s, "accumulated_snapshot")
    screen = dep.training_screen(ctx)
    view = dict(
        ctx, initial=snap, prior=None, store=NamespaceStore(s), packs=dict(train=screen)
    )
    train = dep.profiles(view, "train")
    rule = dict(
        ctx["protocol"], minimum_target_gain=ctx["protocol"]["proposal_score_scale"]
    )
    pool = dep.candidate_pool(train, snap["spec"]["targets"], rule)
    graph = dep.build_graph(view, pool, screen) if pool else None
    ranked = dep.graph_rank(pool, graph, rule) if graph else []
    out = dict(
        weights_sha=accum.weight_sha(snap),
        profile_images=len(screen["ids"]),
        pool_size=len(pool),
        next_proposals=ranked[: ctx["protocol"]["max_attempts"]],
        new_gradient_updates=0,
        suggested_action="Test another freshly scored exchange pass"
        if ranked
        else "No exchange suggested: diagnose proposal/gating limits before changing mechanism",
        scope="TRAINING-only post-batch analysis. Raw/additive predictions are not accepted edits or proof of benefit. It cannot change this run nominee or open another confirmation.",
    )
    s.save("post_batch_analysis", out)
    s.write("post_batch_analysis.json", out)
    memory.assert_immutable(ctx["stamps"])
    gc.collect()
    return out


def run(ctx, accumulation):
    # No fit calls: both arms share exactly the saved 600-update weights/history.
    out = accum.run(ctx, accumulation)
    if out["total_gradient_updates_including_budget_control"] != 0:
        raise AssertionError("Frozen diagnostic performed training")
    analysis = reanalyze(ctx)
    if len(accumulation["accepted_proposals"]) >= ctx["protocol"]["max_accepted"]:
        stopping_reason = "accepted-exchange cap reached; not evidence of a plateau"
    else:
        stopping_reason = "bounded shortlist exhausted; unsearched alternatives remain"
    if out["accepted"]:
        next_action = "Confirm this frozen recipe on independent images/starts before expanding the exchange loop"
    elif out["frozen_eligible"]:
        next_action = "Training/development gain failed historical confirmation; inspect confirmation margins before retrying any mechanism"
    elif accumulation["accepted_proposals"]:
        next_action = "Training-adopted batch failed development: inspect per-category and gate-displacement losses; post-graph proposals are diagnostic only"
    else:
        next_action = "No training batch adopted: inspect graph ranking, actual target gains and preservation failures before changing the proposal family"
    out = dict(
        out,
        post_batch_analysis=analysis,
        source_checkpoint=START_KEY,
        common_start_updates=600,
        new_gradient_updates=0,
        loop_status=dict(
            stopping_reason=stopping_reason,
            next_action=next_action,
            automatically_retried=False,
            post_analysis_layout="Training-adopted batch; may not pass development/confirmation",
        ),
    )
    ctx["store"].write("summary.json", out)
    return out


def export(ctx):
    path = accum.export(ctx)
    renamed = path.with_name(
        f"agf_mature_regroup_{ctx['store'].run_id[:12]}_results.zip"
    )
    path.replace(renamed)
    return renamed
