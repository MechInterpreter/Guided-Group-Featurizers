"""Train the 4,096-group pilot and component controls from fresh public inputs."""

import argparse
import copy
from pathlib import Path

from historical_inputs import (
    activate,
    audit_ids,
    audit_pack,
    coco_context,
    read,
    sha,
    write,
)


def recipe():
    return dict(
        seeds=[21, 22, 23],
        groups=4096,
        width=12,
        top_k=32,
        cap=0.6,
        warm_updates=7640,
        continuation_updates=3000,
        learning_rate=0.0003,
        kinds=["released_vanilla", "mask_only", "fixed_guided", "agf_g"],
        fixed_weight_policy="Arithmetic mean of adaptive weights used at updates 500/1000/1500/2000/2500/3000",
    )


def context(root, source, code):
    activate(code, "components")
    ctx = coco_context(
        root,
        source,
        code,
        "components",
        __file__,
        dict(recipe(), audit_ids=audit_ids(code)),
    )
    ctx["rule"].update(local_save_every=100, drive_save_every=500)
    ctx["rows"] = []
    ctx["test_ids"] = audit_ids(code)
    ctx["test_stamp"] = read(code / "evidence/pilot4096_test/test_identity.json")
    return ctx


def train(ctx):
    import torch
    import components, scale_run
    from coco_baselines import training
    from coco_coordinated.run import required

    s = ctx["store"]
    rows = []
    weights_dir = ctx["local"] / "verified_weights"
    weights_dir.mkdir(exist_ok=True)
    for seed in [21, 22, 23]:
        for kind in ["agf_g", "released_vanilla"]:
            row = scale_run.run_branch(
                ctx, 4096, 32, seed, kind, 0.6, limit=3000, lr=0.0003
            )
            row["development"] = scale_run.development(ctx, row)["report"]
            rows.append(row)
            state = required(s, row["key"])
            tag = f"s{seed}_{kind}"
            torch.save(state["weights"], weights_dir / (tag + ".pt"))
            write(
                weights_dir / (tag + ".json"),
                dict(
                    streams=[state["sampled_batches_sha"], state["sampled_noise_sha"]]
                ),
            )
    import pilot_inputs

    pilot_inputs.catalog(
        dict(
            complete=True,
            final_test_opened=False,
            seven_seed_started=False,
            models=rows,
        )
    )
    s.save("pilot_models_frozen", dict(complete=True, models=rows))
    policies = {}
    warms = {}
    for seed in [21, 22, 23]:
        row = next(r for r in rows if r["seed"] == seed and r["kind"] == "agf_g")
        state = required(s, row["key"])
        policies[str(seed)] = dict(components.fixed_policy(state["history"]), seed=seed)
        key = f"g4096_k32_s{seed}_mask_warm"
        warm = required(s, key)
        path = ctx["local"] / f"warm_s{seed}.pt"
        torch.save(warm["weights"], path)
        warms[str(seed)] = dict(
            key=key,
            weights_sha=training.weight_sha(warm["weights"]),
            file_sha=sha(path),
            updates=7640,
        )
    ctx.update(
        base_context=dict(ctx, rows=copy.deepcopy(rows)), policies=policies, warms=warms
    )
    s.save("fresh_component_inputs", dict(rows=rows, policies=policies, warms=warms))
    components.train_components(ctx)
    write(
        s.output.parent / "FITS_COMPLETE.json",
        dict(
            complete=True,
            models=components.verify_freeze(ctx)["models"],
            store_root=str(s.root),
        ),
    )


def evaluate(ctx, root, code):
    import components, pilot_inputs
    from unittest.mock import patch
    from coco_coordinated.run import required
    from coco_baselines import training

    saved = required(ctx["store"], "fresh_component_inputs")
    ctx.update(rows=saved["rows"], policies=saved["policies"], warms=saved["warms"])
    # Restore disposable inference snapshots from this run's own checkpoints.
    import torch

    folder = ctx["local"] / "verified_weights"
    folder.mkdir(exist_ok=True)
    for row in saved["rows"]:
        state = required(ctx["store"], row["key"])
        if (
            training.weight_sha(state["weights"]) != row["weights_sha"]
            or state["updates"] != 3000
        ):
            raise ValueError("Fresh pilot endpoint changed")
        tag = f"s{row['seed']}_{row['kind']}"
        torch.save(state["weights"], folder / (tag + ".pt"))
    ctx["base_context"] = dict(ctx, rows=saved["rows"])
    complete = components.verify_freeze(ctx)
    pack = audit_pack(ctx, code, root, "controls", complete)
    # The historical tensor hash is reported separately; this run owns new tensors.
    ctx["test_stamp"] = dict(ids=pack["ids"], fresh_identity=pack["identity"])
    ctx["base_context"]["test_stamp"] = ctx["test_stamp"]
    pilot_inputs.score_models(ctx, pack)
    pilot_inputs.paired_results(ctx, required(ctx["store"], "point_metrics_complete"))
    ctx["controls"] = ctx["store"]
    with patch.object(pilot_inputs, "load_cached_test", return_value=pack):
        result = components.evaluate_all(ctx)
    write(
        root / "RESULTS_COMPLETE.json",
        dict(
            complete=True,
            results=str(ctx["store"].results),
            models=complete["models"],
            audit_identity=pack["identity"],
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--features-root", required=True, type=Path)
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
    ctx = context(root, source, code)
    train(ctx) if a.phase == "train" else evaluate(ctx, root, code)


if __name__ == "__main__":
    main()
