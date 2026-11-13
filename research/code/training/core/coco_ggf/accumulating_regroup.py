"""Accumulate strictly positive, dependency-guided, capacity-preserving exchanges.

COCO masks supervise this component. Frozen training-screen adoption is distinct
from development eligibility and historical confirmation. Never open final test.
"""

import copy
import csv
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from coco_adaptation import inputs
from coco_baselines import metrics, training
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import dependency_regroup as dep, direction_regroup as prior
from . import bank_experiment as bank, owner_routing as old, study

DEPENDENCY_RUN = "f4771acac892d3e969e1dce066b9eb66498970da7438a3d5c7539c1dbb47a012"
DEPENDENCY_SHA = "e6da05088e76c3318d3b609f1a670a9aa8b40adcee843a58a22472a6621a3184"


def settings(recovery=True):
    cfg = dep.settings()
    cfg.update(
        version=1,
        study="positive_dependency_exchange_accumulation",
        minimum_target_gain=0.0,
        minimum_mean_gain=0.0,
        positive_epsilon=1e-10,
        run_recovery=bool(recovery),
        max_attempts=8,
        max_accepted=5,
        scope="One saved supervised GGF seed. Training-only accumulation; reused historical development and disjoint historical confirmation. Final test closed.",
    )
    return cfg


def weight_sha(snap):
    return training.weight_sha(snap["weights"])


def prepare(
    parent,
    dependency_output,
    output,
    local,
    sha,
    versions,
    device,
    require_drive=True,
    recovery=True,
):
    cfg = settings(recovery)
    output = Path(output)
    local = Path(local)
    dependency_output = Path(dependency_output)
    a = output.resolve()
    b = dependency_output.resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError(
            "New output must be separate from the read-only dependency parent"
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
    ctx["protocol"] = cfg
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
    w, h, t, row = old.source_start(ctx, cfg["seed"])
    ctx["initial"] = bank.native_snapshot(ctx, w, t, h, 1024, 3, 16)
    source = memory.FrozenStore(
        dependency_output / "checkpoints" / DEPENDENCY_RUN[:20],
        local / "read_dependency",
    )
    if source.run_id != DEPENDENCY_RUN or source.config["source_sha"] != DEPENDENCY_SHA:
        raise ValueError("Wrong dependency parent; no retraining fallback")
    roles = json.loads((source.results / "split_roles.json").read_text())
    if (
        roles["source_weights_sha"] != row["weights_sha"]
        or roles["category_readouts"] != t
    ):
        raise ValueError("Dependency parent weights/readouts changed")
    for split in ("train", "select"):
        if roles["ids"][split] != ctx["packs"][split]["ids"]:
            raise ValueError("Dependency parent image identities changed")
    screen = memory.required(source, "dependency_screen")
    if not screen["rows"]:
        raise ValueError("No saved dependency proposals")
    ctx["dependency"] = source
    ctx["saved_screen"] = screen
    ctx["source_weights_sha"] = row["weights_sha"]
    ctx["stamps"] += memory.source_snapshot([source])
    ctx["store"] = inputs.JournalStore(
        output,
        dict(
            config=cfg,
            source_sha=sha,
            versions=versions,
            device=device,
            parent_run=ctx["source"].run_id,
            activation_run=ctx["original"].run_id,
            dependency_run=DEPENDENCY_RUN,
        ),
        local / "accumulation_rescue",
        require_drive,
    )
    s = ctx["store"]
    s.write(
        "split_roles.json",
        dict(
            ids={k: p["ids"] for k, p in ctx["packs"].items()},
            category_readouts=t,
            source_weights_sha=row["weights_sha"],
            train="Historical graph shortlist and training-screen priority; cumulative exchanges remeasured now. No development reranking.",
            select="Historical 400 development images; one frozen batch then optional recovery checkpoint selection.",
            confirm="One frozen nominee only, against original and appropriate continued-training controls. Historical, not new or seed-independent.",
            final_test_opened=False,
        ),
    )
    s.event(
        "prepared",
        historical_numerical_versions=source.config["versions"],
        current_versions=versions,
        all_metrics_rescored=True,
        minimum_mean_gain=0.0,
        minimum_target_gain=0.0,
    )
    return ctx


def _categories(report):
    rows = report["categories"]
    out = {r["category_index"]: r for r in rows}
    if len(out) != len(rows):
        raise ValueError("Duplicate category scores")
    return out


def quality_gate(report, refs, cfg):
    """Guard cumulative damage against roots, not just the most recent step."""
    q = bank.quality(report, refs, cfg)
    a = _categories(report)
    for name, ref in refs.items():
        b = _categories(ref)
        if set(a) != set(b):
            raise ValueError("Category cohort changed")
        for c in a:
            for metric in ("capture", "ap"):
                q["margins"][f"{name}_category_{c}_{metric}"] = (
                    a[c][metric] - b[c][metric] + cfg["max_category_drop"]
                )
    q["failures"] = [
        k for k, v in q["margins"].items() if not np.isfinite(v) or v < -1e-12
    ]
    q["passed"] = not q["failures"]
    return q


def positive_gate(report, current, root, cfg, target=None):
    q = quality_gate(report, dict(current=current, cumulative_root=root), cfg)
    gains = dict(mean_capture=report["macro_capture"] - current["macro_capture"])
    if target is not None:
        gains["target_capture"] = (
            _categories(report)[target]["capture"]
            - _categories(current)[target]["capture"]
        )
    for name, gain in gains.items():
        q["margins"]["positive_" + name] = gain - cfg["positive_epsilon"]
        if not np.isfinite(gain) or gain <= cfg["positive_epsilon"]:
            q["failures"].append("positive_" + name)
    q["passed"] = not q["failures"]
    q["gains"] = gains
    return q


def proposal_order(saved, cfg):
    # Every graph-shortlisted exchange is reconsidered, including old rejections.
    # Priority comes only from the already recorded TRAINING screen, not select.
    rows = [r for r in saved["rows"] if r["in_graph_shortlist"]]
    rows.sort(
        key=lambda r: (
            -r["mean_capture_delta"],
            -r["target_capture_delta"],
            r["proposal"]["id"],
        )
    )
    return [copy.deepcopy(r["proposal"]) for r in rows[: cfg["max_attempts"]]]


def conflict(proposal, used_groups, used_targets):
    if proposal["category_index"] in used_targets:
        return "target_already_accepted"
    groups = {proposal["receiver_group"], proposal["donor_group"]}
    if groups & used_groups:
        return "group_already_changed"
    return None


def category_deltas(report, ref):
    a = _categories(report)
    b = _categories(ref)
    if set(a) != set(b):
        raise ValueError("Category cohort changed")
    return [
        dict(
            category_index=c,
            category=a[c].get("category", str(c)),
            capture_delta=a[c]["capture"] - b[c]["capture"],
            ap_delta=a[c]["ap"] - b[c]["ap"],
        )
        for c in sorted(a)
    ]


def measured(ctx, key, snap, split):
    s = ctx["store"]
    sha = weight_sha(snap)
    cached = s.load(key)
    signature = identity(
        dict(
            weight_sha=sha,
            spec=snap["spec"],
            ids=ctx["packs"][split]["ids"],
            split=split,
        )
    )
    if cached is not None:
        if cached["signature"] != signature:
            raise ValueError("Cached report model or images changed: " + key)
        return cached["report"]
    s.event("score_start", key=key, split=split, images=len(ctx["packs"][split]["ids"]))
    start = time.perf_counter()
    m = prior.model(ctx, snap)
    report = bank.evaluate(ctx, m, split)
    del m
    gc.collect()
    s.save(
        key,
        dict(signature=signature, report=report, seconds=time.perf_counter() - start),
    )
    s.event(
        "score_complete",
        key=key,
        split=split,
        seconds=time.perf_counter() - start,
        **bank.scalar(report),
    )
    return report


def accumulate(ctx):
    s = ctx["store"]
    done = s.load("accumulation_complete")
    if done is not None:
        return done
    screen = dep.training_screen(ctx)
    ctx["packs"]["train_screen"] = screen
    graph = memory.required(ctx["dependency"], "dependency_graph")
    if graph["image_ids"] != screen["ids"]:
        raise ValueError("Graph screen image identities changed")
    # Historical graph is a proposal prior. Gate diagnostics also use CURRENT
    # runtime predictions, so a numerical-version change is never hidden.
    graph = dict(graph)
    m = prior.model(ctx, ctx["initial"])
    m.eval()
    gates = []
    with torch.no_grad():
        for i, x in enumerate(screen["x"]):
            xx = torch.as_tensor(np.array(x, copy=True), device=ctx["device"])
            gates.append(
                np.packbits((m.encode(xx).norm(dim=-1) > 1e-6).cpu().numpy(), axis=1)
            )
            if (i + 1) % ctx["protocol"]["shard_images"] == 0:
                s.event("original_gate_progress", done=i + 1, total=len(screen["ids"]))
    graph["base_gate_packed"] = np.stack(gates)
    del m
    gc.collect()
    original = ctx["initial"]
    current = original
    root = measured(ctx, "train_root", original, "train_screen")
    report = root
    s.save("accumulation_root", original)
    order = proposal_order(ctx["saved_screen"], ctx["protocol"])
    s.write("proposal_order.json", order)
    steps = []
    accepted = []
    groups = set()
    targets = set()
    for i, p in enumerate(order):
        why = conflict(p, groups, targets)
        if len(accepted) >= ctx["protocol"]["max_accepted"]:
            why = "maximum_accepted_reached"
        key = f"step{i:02d}"
        saved = s.load(key)
        before = weight_sha(current)
        if saved is not None:
            if saved["before_weights_sha"] != before or saved["proposal"] != p:
                raise ValueError("Cumulative resume chain changed")
            row = saved
        elif why:
            row = dict(
                proposal=p,
                accepted=False,
                skip=why,
                before_weights_sha=before,
                after_weights_sha=before,
            )
            s.save(key, row)
        else:
            candidate = prior.permute_snapshot(current, p["outgoing"], p["incoming"])
            # No gradients: all values, heads and readouts preserved, only slots move.
            r = measured(ctx, key + "_score", candidate, "train_screen")
            gate = positive_gate(r, report, root, ctx["protocol"], p["category_index"])
            m = prior.model(ctx, candidate)
            effects = dep.gate_effects(ctx, m, screen, graph, p)
            del m
            gc.collect()
            effects["scope"] = (
                "CUMULATIVE model versus original gates; unrelated-group counts are relative to this proposal pair."
            )
            row = dict(
                proposal=p,
                accepted=gate["passed"],
                skip=None,
                gate=gate,
                report=r,
                before_weights_sha=before,
                after_weights_sha=weight_sha(candidate) if gate["passed"] else before,
                candidate_weights_sha=weight_sha(candidate),
                deltas_vs_current=category_deltas(r, report),
                deltas_vs_root=category_deltas(r, root),
                cumulative_gate_effects=effects,
                snapshot=key + "_accepted" if gate["passed"] else None,
            )
            # Snapshot first, decision last: a crash can safely replay this step.
            if gate["passed"]:
                s.save(row["snapshot"], candidate)
            s.save(key, row)
        if row["accepted"]:
            current = memory.required(s, row["snapshot"])
            if weight_sha(current) != row["after_weights_sha"]:
                raise ValueError("Accepted snapshot changed")
            report = row["report"]
            accepted.append(p)
            groups.update((p["receiver_group"], p["donor_group"]))
            targets.add(p["category_index"])
        else:
            if weight_sha(current) != before:
                raise AssertionError("Rejected exchange changed current weights")
        steps.append(row)
        s.write("steps/" + key + ".json", row)
        s.event(
            "accumulation_step",
            done=i + 1,
            total=len(order),
            proposal=p["id"],
            object=p["category"],
            accepted=row["accepted"],
            accepted_total=len(accepted),
            skip=row.get("skip"),
            gains=row.get("gate", {}).get("gains"),
            failures=row.get("gate", {}).get("failures", []),
        )
    s.save("accumulated_snapshot", current)
    out = dict(
        steps=steps,
        accepted_proposals=accepted,
        initial_report=root,
        final_report=report,
        final_checkpoint="accumulated_snapshot",
        gradient_updates=0,
        final_weight_sha=weight_sha(current),
        final_test_opened=False,
    )
    s.save("accumulation_complete", out)
    s.write("accumulation.json", out)
    del ctx["packs"]["train_screen"]
    del screen
    gc.collect()
    return out


def choose_eligible(rows):
    eligible = [r for r in rows if r["gate"]["passed"]]
    return (
        max(
            eligible,
            key=lambda r: (
                r["report"]["macro_capture"],
                r["report"]["macro_ap"],
                -r["report"]["reconstruction"]["mse"],
            ),
        )
        if eligible
        else None
    )


def paired_audit(ctx, edited, control, split, key):
    s = ctx["store"]
    cached = s.load("paired_" + key)
    if cached is not None:
        return cached
    mapping = dict(
        mappings=[
            dict(
                category_index=t["category_index"],
                detector=dict(group=t["detector_group"]),
                contrast=dict(groups=[t["contrast_group"]]),
            )
            for t in ctx["initial"]["spec"]["targets"]
        ]
    )

    def compact(snap, label):
        m = prior.model(ctx, snap)
        p = ctx["packs"][split]
        audit = metrics.audit_arrays(
            m, p["x"], p["masks"], p["ids"], ctx["cfg"], ctx["device"]
        )
        a = study.compact_audit(audit, mapping, ctx["cfg"])
        del m, audit
        gc.collect()
        folder = s.results / "image_audits"
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / (key + "_" + label + ".npz"),
            **{k: v for k, v in a.items() if isinstance(v, np.ndarray)},
            image_ids=np.array(a["image_ids"]),
            category_indices=np.array(a["category_indices"]),
            annotation_ids=np.array(a["objects"]),
        )
        return a

    a = compact(edited, "edited")
    b = compact(control, "control")
    ci = study.paired(
        a,
        b,
        dict(
            ctx["cfg"], bootstrap=ctx["protocol"]["bootstrap"], fresh_test_seed=105071
        ),
    )
    ci.update(
        scope="Paired whole-image uncertainty conditional on fixed weights. Historical images; not selection-corrected or independent-seed evidence.",
        split=split,
    )
    s.save("paired_" + key, ci)
    s.write("paired/" + key + ".json", ci)
    return ci


def run(ctx, accumulation):
    s = ctx["store"]
    done = s.load("experiment_complete")
    if done is not None:
        return done
    cfg = ctx["protocol"]
    initial = ctx["initial"]
    snap = memory.required(s, "accumulated_snapshot")
    base = measured(ctx, "select_original", initial, "select")
    frozen = measured(ctx, "select_accumulated", snap, "select")
    gate = positive_gate(frozen, base, base, cfg)
    s.write(
        "development/frozen.json",
        dict(report=frozen, gate=gate, deltas=category_deltas(frozen, base)),
    )
    frozen_eligible = bool(accumulation["accepted_proposals"]) and gate["passed"]
    candidates = (
        [
            dict(
                checkpoint="accumulated_snapshot",
                control="accumulation_root",
                update=0,
                report=frozen,
                gate=gate,
            )
        ]
        if frozen_eligible
        else []
    )
    training_checks = []
    budget_history = None
    used = 0
    if frozen_eligible and cfg["run_recovery"]:
        # One batch, one ordinary matched no-edit control. All failed updates count.
        control = prior.fit(ctx, "continued_ggf", initial, affected=None)
        edit = prior.fit(ctx, "batch_recovery", snap, affected=None)
        if edit["batch_sha"] != control["batch_sha"]:
            raise AssertionError("Paired recovery samples/noise differ")
        controls = {r["update"]: r for r in control["history"]}
        for e in edit["history"]:
            if e["update"] == 0:
                continue
            c = controls[e["update"]]
            # Shared training drift is separately reported, rather than charged
            # only to the edit. Quality and per-object safeguards vs matched control.
            g = positive_gate(e["report"], c["report"], c["report"], cfg)
            absolute = quality_gate(e["report"], dict(original=base), cfg)
            row = dict(
                checkpoint=e["checkpoint"],
                control=c["checkpoint"],
                update=e["update"],
                report=e["report"],
                gate=g,
                absolute_original_guard=absolute,
                edit_deltas_vs_control=category_deltas(e["report"], c["report"]),
                shared_control_drift=category_deltas(c["report"], base),
            )
            training_checks.append(row)
            if g["passed"]:
                candidates.append(row)
        used = 2 * cfg["recovery_updates"]
        # A stricter total-gradient-budget comparison accounts for candidate AND
        # no-edit control training. Frozen phase has zero search gradient updates.
        budget = prior.fit(
            ctx, "search_budget_continued_ggf", initial, affected=None, updates=used
        )
        budget_history = budget["history"]
        strong = max(
            budget_history,
            key=lambda r: (
                r["report"]["macro_capture"],
                r["report"]["macro_ap"],
                -r["report"]["reconstruction"]["mse"],
            ),
        )
        for row in candidates:
            row["matched_update_gate"] = row["gate"]
            row["gate"] = positive_gate(
                row["report"], strong["report"], strong["report"], cfg
            )
            row["budget_control"] = strong["checkpoint"]
    nominee = choose_eligible(candidates)
    # Freeze once, before loading confirmation; never retry a different candidate.
    nomination = dict(
        nominee=nominee,
        checks=training_checks,
        frozen_eligible=frozen_eligible,
        search_gradient_updates=used,
        total_gradient_updates_including_budget_control=2 * used,
        budget_control_history=budget_history,
    )
    prior_nomination = s.load("frozen_nomination")
    if prior_nomination is not None and identity(prior_nomination) != identity(
        nomination
    ):
        raise ValueError("Cannot change nominee after freezing confirmation protocol")
    if prior_nomination is None:
        s.save("frozen_nomination", nomination)
    s.write("development/nomination.json", nomination)
    paired_dev = paired_audit(ctx, snap, initial, "select", "frozen_development")
    confirmation = None
    accepted = False
    if nominee:
        norm = memory.required(ctx["original"], "normalization")
        ctx["packs"]["confirm"] = memory.normalized_pack(
            ctx["original"], "confirm", ctx["provenance"], norm, ctx["local"], 20
        )
        sets = [set(p["ids"]) for p in ctx["packs"].values()]
        if any(sets[i] & sets[j] for i in range(len(sets)) for j in range(i)):
            raise ValueError("Confirmation image overlap")
        chosen = memory.required(s, nominee["checkpoint"])
        control_snap = memory.required(
            s, nominee.get("budget_control", nominee["control"])
        )
        a = measured(ctx, "confirm_nominee", chosen, "confirm")
        b = measured(ctx, "confirm_control", control_snap, "confirm")
        g = positive_gate(a, b, b, cfg)
        ci = paired_audit(ctx, chosen, control_snap, "confirm", "nominee_confirmation")
        confirmation = dict(
            edited=a,
            control=b,
            gate=g,
            paired=ci,
            scope="Historical disjoint confirmation; not untouched final test. No candidate retry.",
            mean_capture_interval_supports_positive=bool(
                ci["capture_gain_ci95"][0] > 0
            ),
        )
        accepted = g["passed"]
        s.write("confirmation.json", confirmation)
    out = dict(
        complete=True,
        accepted=accepted,
        training_accepted_exchanges=len(accumulation["accepted_proposals"]),
        accepted_proposals=accumulation["accepted_proposals"],
        original=base,
        accumulated=frozen,
        frozen_gate=gate,
        frozen_eligible=frozen_eligible,
        training_checks=training_checks,
        nominee=nominee,
        search_gradient_updates=used,
        total_gradient_updates_including_budget_control=2 * used,
        budget_control_history=budget_history,
        confirmation=confirmation,
        paired_development=paired_dev,
        final_test_opened=False,
        scope=cfg["scope"],
        interpretation="Small positive point estimates can be adopted under safeguards. Adoption is not statistical proof or a multi-seed overall win.",
    )
    s.save("experiment_complete", out)
    s.write("summary.json", out)
    memory.assert_immutable(ctx["stamps"])
    plots(ctx, out, accumulation)
    return out


def plots(ctx, out, accumulation):
    s = ctx["store"]
    rows = [
        dict(method="Saved GGF", **bank.scalar(out["original"])),
        dict(method="Frozen accumulated exchanges", **bank.scalar(out["accumulated"])),
    ]
    if out["budget_control_history"]:
        best = max(
            out["budget_control_history"],
            key=lambda r: (
                r["report"]["macro_capture"],
                r["report"]["macro_ap"],
                -r["report"]["reconstruction"]["mse"],
            ),
        )
        rows.append(
            dict(method="Best total-search-budget GGF", **bank.scalar(best["report"]))
        )
    if out["nominee"]:
        rows.append(
            dict(method="Frozen nominee", **bank.scalar(out["nominee"]["report"]))
        )
    with (s.results / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ("capture", "ap", "ve")):
        ax.barh(range(len(rows)), [r[metric] * 100 for r in rows])
        ax.set_yticks(range(len(rows)), [r["method"] for r in rows])
        ax.invert_yaxis()
        ax.set_title(metric.upper() + " (%)")
    fig.suptitle(
        "Historical development: point estimates; final acceptance reported separately"
    )
    fig.tight_layout()
    (s.results / "figures").mkdir(exist_ok=True)
    fig.savefig(s.results / "figures/comparison.png", dpi=150)
    plt.close(fig)
    values = [accumulation["initial_report"]["macro_capture"]]
    labels = ["start"]
    for step in accumulation["steps"]:
        if step["accepted"]:
            values.append(step["report"]["macro_capture"])
            labels.append(step["proposal"]["category"])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(range(len(values)), np.array(values) * 100, "o-")
    ax.set_xticks(range(len(values)), labels, rotation=20)
    ax.set_ylabel("Mean capture (%)")
    ax.set_title("Accepted frozen exchanges: training screen only")
    fig.tight_layout()
    fig.savefig(s.results / "figures/accumulation.png", dpi=150)
    plt.close(fig)
    return rows


def export(ctx):
    s = ctx["store"]
    path = s.output / f"agf_accumulating_regroup_{s.run_id[:12]}_results.zip"
    manifest = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(s.results.rglob("*")):
            if p.is_file():
                name = "results/" + p.relative_to(s.results).as_posix()
                z.write(p, name)
                manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        z.writestr("artifact_hashes.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise OSError("Export corruption")
    return path
