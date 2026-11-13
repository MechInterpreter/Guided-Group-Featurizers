"""Run the complete matched 12D study and post-run TRAIN rank diagnostics."""

import argparse, gc, json, os, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    options = json.loads(Path(args.config).read_text())
    source = Path(__file__).resolve().parent
    local = Path(options["local"])
    local.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    marker = local / "twelve_selftest.json"
    if (
        not marker.exists()
        or json.loads(marker.read_text()).get("source_sha") != options["source_sha"]
    ):
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(source / "tests/test_twelve_capacity.py"),
                str(source / "test_rank.py"),
                str(source / "tests/test_twelve_rank.py"),
                str(source / "tests/test_twelve_export.py"),
                "--disable-warnings",
                "--basetemp",
                str(local / ("selftest_" + os.urandom(6).hex())),
            ],
            cwd=source,
            env=env,
        )
        marker.write_text(
            json.dumps(dict(source_sha=options["source_sha"], passed=True))
        )
    import torch
    from threadpoolctl import threadpool_limits
    from coco_ggf import twelve_capacity as exp, twelve_rank as rank

    if exp.matched.versions() != options["expected_versions"]:
        raise RuntimeError("Numerical packages differ from the completed source")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("Select a Tesla T4 runtime")
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    ctx = None
    stage = "prepare"
    try:
        print(
            "Reading completed 8D AGF/Vanilla checkpoints. No HF token, extraction or repeated warm-up.",
            flush=True,
        )
        ctx = exp.prepare(
            options["paths"],
            options["output"],
            options["local"],
            options["cache_local"],
            options["source_sha"],
        )
        ctx["store"].results.joinpath("PROTOCOL.md").write_bytes(
            (source / "PROTOCOL_12D.md").read_bytes()
        )
        stage = "verify_saved_rank"
        rank.verify_source(ctx, options["paths"]["rank"])
        stage = "T4_preflight"
        exp.preflight(ctx)
        stage = "training"
        exp.run(ctx, options["source_zip"])
        stage = "test_evaluation"
        result = exp.evaluate(ctx)
        stage = "TRAIN_effective_rank"
        ctx["packs"].pop("test", None)
        gc.collect()
        rank_result = rank.run(ctx, options["paths"]["rank"])
        ctx["store"].write(
            "completion_status.json",
            dict(
                training_complete=True,
                test_complete=result["complete"],
                rank_complete=rank_result["complete"],
                new_planned_updates=36000,
            ),
        )
        stage = "export"
        archive = exp.export(ctx, options["source_zip"])
        receipt = dict(
            complete=result["complete"] and rank_result["complete"],
            source_sha=options["source_sha"],
            parent_run=exp.PARENT_RUN,
            result_zip=str(archive),
            results=str(ctx["store"].results),
            new_planned_gradient_updates=36000,
            new_endpoints=12,
            reused_source_models=6,
            rank_complete=True,
            new_models_ranked=12,
            saved_8d_ranks_reused=6,
        )
        Path(options["receipt"]).write_text(json.dumps(receipt, indent=2))
        print("12D MATCHED STUDY AND EFFECTIVE RANK COMPLETE:", archive, flush=True)
    except BaseException as exc:
        if ctx is not None:
            exp.failure(ctx, stage, exc, options["source_zip"])
        raise
    finally:
        if ctx is not None:
            ctx["packs"].clear()
            gc.collect()


if __name__ == "__main__":
    main()
