"""Restore original 4D audits from the completed, pinned continuation source."""

from coco_baselines.storage import identity
from coco_coordinated import run as memory
from . import next_experiments as prior


def restore(ctx):
    source = ctx["sources"]["previous"]
    if source.run_id != prior.PREVIOUS_RUN:
        raise ValueError("Wrong completed continuation source")
    summary = memory.required(source, "experiment_complete")
    if not summary.get("complete"):
        raise ValueError("Previous continuation evaluation is incomplete")
    pending = []
    for seed in prior.SEEDS:
        for arm in ["cap040", "vanilla4"]:
            row = prior.root_for(ctx, seed, arm)
            tag = f"s{seed}_{arm}"
            saved = memory.required(source, "audit_" + tag)
            compact = memory.required(source, "compact_" + tag)
            if saved["identity"] != prior.base.report_signature(
                row, ctx["test_stamp"], ctx["cfg"]
            ):
                raise ValueError(
                    "Saved 4D audit model, readout, test or scoring protocol differs: "
                    + tag
                )
            prior.base.check_report(saved["report"], row, ctx["cfg"])
            entries = [
                r for r in summary["models"] if (r["seed"], r["arm"]) == (seed, arm)
            ]
            if len(entries) != 1 or identity(entries[0]) != identity(saved["report"]):
                raise ValueError(
                    "Completed summary disagrees with saved 4D audit: " + tag
                )
            if (
                compact["image_ids"] != ctx["test_ids"]
                or compact["category_indices"] != ctx["cfg"]["category_indices"]
            ):
                raise ValueError("Saved 4D compact audit cohort differs: " + tag)
            for prefix, value in [("audit_", saved), ("compact_", compact)]:
                key = prefix + tag
                old = ctx["store"].load(key)
                if old is not None and identity(
                    prior.engine.state_identity(old)
                ) != identity(prior.engine.state_identity(value)):
                    raise ValueError(
                        "Existing destination audit disagrees with immutable source: "
                        + key
                    )
                pending.append(
                    (key, value, prior.matched.source_key_stamp(source, key))
                )
    # Validate every source and destination before writing any recovered pair.
    prior.matched.assert_sources(ctx)
    for key, value, stamp in pending:
        if ctx["store"].load(key) is None:
            ctx["store"].save(key, value)
        if stamp not in ctx["checkpoint_stamps"]:
            ctx["checkpoint_stamps"].append(stamp)
    record = dict(
        recovery="reuse_original_4D_audits_from_completed_continuation",
        source_run=source.run_id,
        restored_keys=[key for key, _, _ in pending],
        new_gradient_updates=0,
        new_inference=False,
    )
    ctx["store"].write("control_audit_reuse.json", record)
    ctx["store"].write("source_checkpoint_hashes.json", ctx["checkpoint_stamps"])
    prior.matched.assert_sources(ctx)
    print(
        "Verified and reused six saved 4D control audits; no retraining or inference.",
        flush=True,
    )
    return record
