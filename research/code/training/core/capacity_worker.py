"""Fresh T4 process for all warm-start capacity branches and final audits."""

import argparse, gc, json, os, subprocess, sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    a = p.parse_args()
    options = json.loads(Path(a.config).read_text())
    source = Path(__file__).resolve().parent
    local = Path(options["local"])
    local.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        PYTHONPATH=str(source) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    marker = local / "capacity_selftest.json"
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
                str(source / "tests/test_capacity_sweep.py"),
                str(source / "tests/test_reserve_additions.py"),
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
    from coco_ggf import warm_capacity as exp
    from coco_ggf import reserve_additions as reserve, capacity_reuse

    if exp.matched.versions() != options["expected_versions"]:
        raise RuntimeError("Worker packages differ from saved source")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("Select Tesla T4")
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    context = None
    reserve_context = None
    stage = "prepare"
    try:
        print(
            "Reading completed ordinary 5D AGF and Vanilla checkpoints; no HF token or extraction.",
            flush=True,
        )
        context = exp.prepare(
            options["paths"],
            options["output"],
            options["local"],
            options["cache_local"],
            options["source_sha"],
        )
        context["store"].results.joinpath("PROTOCOL.md").write_bytes(
            (source / "PROTOCOL_CAPACITY.md").read_bytes()
        )
        print("Verified Drive checkpoint folder:", context["store"].root, flush=True)
        stage = "reserve_additions_prepare"
        reserve_context = reserve.prepare(
            context, options["reserve_output"], options["source_sha"]
        )
        stage = "reserve_additions_prescreen"
        reserve_result = reserve.run(reserve_context, options["source_zip"])
        reserve_archive = reserve.export(reserve_context, options["source_zip"])
        print(
            "First-stage development evidence ZIP saved:", reserve_archive, flush=True
        )
        # Keep both phases in the final compact evidence ZIP while retaining
        # separate resumable checkpoint stores and manifest identities.
        for p in reserve_context["store"].results.rglob("*"):
            if p.is_file():
                dest = (
                    context["store"].results
                    / "reserve_additions"
                    / p.relative_to(reserve_context["store"].results)
                )
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(p.read_bytes())
        context["store"].write(
            "combined_status.json",
            dict(
                reserve_prescreen_complete=True,
                reserve_results=str(reserve_context["store"].results),
                reserve_zip=str(reserve_archive),
                capacity_complete=False,
            ),
        )
        stage = "restore_previous_capacity_progress"
        capacity_reuse.restore(context)
        stage = "T4_preflight"
        exp.preflight(context)
        stage = "capacity_training"
        exp.run(context, options["source_zip"])
        stage = "test_evaluation"
        result = exp.evaluate(context)
        stage = "export"
        context["store"].write(
            "combined_status.json",
            dict(
                reserve_prescreen_complete=True,
                reserve_results=str(reserve_context["store"].results),
                reserve_zip=str(reserve_archive),
                capacity_complete=True,
            ),
        )
        archive = exp.export(context, options["source_zip"])
        receipt = dict(
            complete=result["complete"],
            source_sha=options["source_sha"],
            parent_run=exp.PARENT_RUN,
            result_zip=str(archive),
            results=str(context["store"].results),
            new_planned_gradient_updates=72000,
            reserve_prescreen_complete=reserve_result["complete"],
            reserve_planned_gradient_updates=6000,
            combined_planned_gradient_updates=78000,
            reserve_results=str(reserve_context["store"].results),
            reserve_zip=str(reserve_archive),
        )
        Path(options["receipt"]).write_text(json.dumps(receipt, indent=2))
        print("ALL 24 CAPACITY ENDPOINTS COMPLETE:", archive, flush=True)
    except BaseException as exc:
        if reserve_context is not None and stage.startswith("reserve_"):
            reserve.failure(reserve_context, stage, exc, options["source_zip"])
        elif context is not None:
            exp.failure(context, stage, exc, options["source_zip"])
        raise
    finally:
        if context is not None:
            context["packs"].clear()
            gc.collect()
        if reserve_context is not None:
            reserve_context["packs"].clear()
            gc.collect()


if __name__ == "__main__":
    main()
