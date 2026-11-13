"""Read-only import of progress from the immediately preceding notebook version."""

import copy, json
from pathlib import Path
from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import warm_capacity as cap

PREVIOUS_SHA = "08f980a9465737be9c7df2b572f85d8104ea3802602894228b3ff00f9e5da9ae"


def restore(ctx):
    s = ctx["store"]
    candidate = dict(s.config, source_sha=PREVIOUS_SHA)
    run = identity(candidate)
    manifest = s.output / "results" / run[:20] / "manifest.json"
    if run == s.run_id or not manifest.exists():
        return dict(reused=False, reason="No exact compatible previous capacity run")
    meta = json.loads(manifest.read_text())
    if (
        meta["run_id"] != run
        or identity(meta["config"]) != run
        or meta["config"] != candidate
    ):
        raise ValueError("Previous capacity manifest differs")
    old = cap.matched.discover(
        s.output, ctx["local"] / "read_previous_capacity", run=run
    )
    keys = [f"{arm}_s{seed}" for arm in cap.ARMS for seed in cap.SEEDS]
    keys += [
        "capacity_preflight",
        "development_endpoints",
        "capacity_frozen",
        "capacity_result",
    ]
    for seed in cap.SEEDS:
        for arm in cap.ARMS + cap.SOURCE_ARMS:
            keys += [f"audit_s{seed}_{arm}", f"compact_s{seed}_{arm}"]
        keys += [f"paired_s{seed}_{a}_vs_{b}" for a, b in cap.evaluation_pairs()]
    imported = []
    for key in keys:
        value = old.load(key)
        if value is None:
            continue
        present = s.load(key)
        if present is None:
            s.save(key, value)
            imported.append(key)
            ctx["checkpoint_stamps"].append(cap.matched.source_key_stamp(old, key))
        # A resumed new run may already be farther ahead; do not overwrite it.
        del value, present
    frozen = s.load("capacity_frozen")
    if frozen is not None:
        cap.verify_freeze(ctx)
    for p in old.results.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(old.results)
        name = rel.as_posix()
        if rel.parts[0] not in [
            "histories",
            "development",
            "reports",
            "figures",
        ] and not name.startswith(("development_", "test_")):
            continue
        dest = s.results / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    report = dict(
        reused=bool(imported),
        source_run=run,
        source_sha=PREVIOUS_SHA,
        imported_keys=imported,
        source_read_only=True,
        rule="Exact full config match except the known packaging SHA. Numerical capacity implementation and fit fingerprints remain unchanged.",
    )
    s.write("previous_capacity_reuse.json", report)
    ctx["sources"]["previous_capacity"] = old
    ctx["stamps"] += memory.source_snapshot([old])
    s.write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    print(
        "Imported verified previous capacity progress:",
        len(imported),
        "artifacts; completed fits and audits will be reused.",
        flush=True,
    )
    return report
