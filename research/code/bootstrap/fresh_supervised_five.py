"""Fresh five-order supervised fits and their separately reserved holdout."""

import argparse
import copy
from pathlib import Path

from historical_inputs import (
    activate,
    audit_pack,
    coco_context,
    InputView,
    open_study,
    read,
    separate,
    write,
)


def install_precision(store):
    import supervised_study as study
    import object_schedule
    from training_precision import patch_paths, JointPrecisionRetry
    from precision_retry import PrecisionRetry
    from coco_ggf.slics_repair import Solver

    partial = PrecisionRetry(study.partial_nnls)
    joint = JointPrecisionRetry(Solver.solve)
    patch_paths(study, object_schedule)

    def solve(self, x, allowed=None, force_fallback=False):
        return joint(self, x, allowed, force_fallback)

    Solver.solve = solve
    study.partial_nnls = partial
    partial.store = joint.store = store


def context(root, source, code, component_root):
    separate(root, source, code, component_root)
    activate(code, "supervised_five_seed")
    import supervised_study as study
    import five_seed
    from coco_baselines.storage import identity
    from coco_baselines import training
    from coco_coordinated.run import required

    protocol = five_seed.evaluation_protocol(study.protocol())
    plan = read(code / "evidence/supervised_five_seed/fresh_holdout_plan.json")
    if identity(plan) != study.protocol()["holdout_reservation_sha"]:
        raise ValueError("Recorded annotation-only holdout reservation changed")
    ctx = coco_context(
        root,
        source,
        code,
        "supervised_five_seed",
        __file__,
        dict(protocol, fresh_anchor_run=read(component_root / "FRESH_STUDY.json")),
    )
    parent = open_study(component_root, ctx["local"] / "anchor_source")
    done = read(component_root / "FITS_COMPLETE.json")
    if not done.get("complete"):
        raise ValueError("Complete the fresh 4096-group fits first")
    rows = [r for r in done["models"] if r["kind"] == "agf_g"]
    if len(rows) != 3 or {r["seed"] for r in rows} != {21, 22, 23}:
        raise ValueError("Retain all three 4096-group capacity anchors")
    if parent.config["training_identity"] != ctx["packs"]["train"][
        "identity"
    ] or parent.config["normalization_sha"] != identity(ctx["norm"]):
        raise ValueError("Capacity-anchor data or normalization differ")
    for row in rows:
        state = required(parent, row["key"])
        if (
            state["updates"] != 3000
            or row["groups"] != 4096
            or training.weight_sha(state["weights"]) != row["weights_sha"]
        ):
            raise ValueError("Fresh capacity-anchor checkpoint changed")
    ctx["store"] = InputView(ctx["store"], {r["key"]: parent for r in rows})
    ctx.update(pilot_rows=rows, core_rule=copy.deepcopy(ctx["rule"]))
    ctx["rule"].update(fresh_seed=126109, final_holdout_images=1500)
    ctx["store"].save("fresh_holdout_plan", plan)
    ctx["store"].write("fresh_holdout_plan.json", plan)
    install_precision(ctx["store"])
    return ctx, protocol


def train(root, ctx, protocol):
    import supervised_study as study
    import five_seed

    choice = study.selection(ctx)
    rows = []
    for seed in five_seed.SEEDS:
        rows.extend(study.final_rows(ctx, seed, choice))
    five_seed.validate_rows(rows)
    five_seed.install_evaluation_scope(rows, protocol)
    study.freeze(ctx, choice)
    complete = study.verify_frozen(ctx)
    write(
        root / "FITS_COMPLETE.json",
        dict(
            complete=True,
            models=rows,
            protocol=protocol,
            store_root=str(ctx["store"].root),
            settings=choice,
        ),
    )


def evaluate(root, ctx, code, protocol):
    import supervised_study as study
    import five_seed

    done = read(root / "FITS_COMPLETE.json")
    if not done.get("complete") or done["protocol"] != protocol:
        raise ValueError("Complete the unchanged five-order catalog first")
    five_seed.validate_rows(done["models"])
    five_seed.install_evaluation_scope(done["models"], protocol)
    frozen = study.verify_frozen(ctx)
    pack = audit_pack(ctx, code, root, "supervised-five", frozen)
    result = study.audit(ctx, pack)
    write(
        root / "RESULTS_COMPLETE.json",
        dict(
            complete=True,
            results=str(ctx["store"].results),
            audit_identity=pack["identity"],
            models=done["models"],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--features-root", required=True, type=Path)
    p.add_argument("--components-root", required=True, type=Path)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--phase", choices=["train", "evaluate"], required=True)
    a = p.parse_args()
    root, source, code = (
        a.root.resolve(),
        a.features_root.resolve(),
        a.code_root.resolve(),
    )
    ctx, protocol = context(root, source, code, a.components_root.resolve())
    train(root, ctx, protocol) if a.phase == "train" else evaluate(
        root, ctx, code, protocol
    )


if __name__ == "__main__":
    main()
