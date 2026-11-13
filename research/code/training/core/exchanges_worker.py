"""Fresh numerical worker: one 5D exchange pass per seed, then cached-test audit."""

import argparse, gc, json, os, subprocess, sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
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
    marker = local / "selftest_passed.json"
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
                str(source / "tests/test_five_exchanges.py"),
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
    from coco_ggf import five_exchanges as exp

    if exp.matched.versions() != options["expected_versions"]:
        raise RuntimeError("Worker versions differ from saved numerical source")
    if not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name():
        raise RuntimeError("Select Tesla T4")
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    context = None
    stage = "prepare"
    try:
        print(
            "Reading final ordinary 5D checkpoints; no optimizer or feature extraction will run.",
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
            (source / "PROTOCOL_EXCHANGES.md").read_bytes()
        )
        print("Verified Drive checkpoints:", context["store"].root, flush=True)
        stage = "dependency_profiles_and_exchanges"
        exp.run(context, options["source_zip"])
        stage = "test_evaluation"
        result = exp.evaluate(context)
        stage = "export"
        archive = exp.export(context, options["source_zip"])
        receipt = dict(
            complete=result["complete"],
            source_sha=options["source_sha"],
            parent_run=exp.PARENT_RUN,
            result_zip=str(archive),
            results=str(context["store"].results),
            new_gradient_updates=0,
        )
        Path(options["receipt"]).write_text(json.dumps(receipt, indent=2))
        print("ALL THREE SEEDS COMPLETE:", archive, flush=True)
    except BaseException as exc:
        if context is not None:
            exp.failure(context, stage, exc, options["source_zip"])
        raise
    finally:
        if context is not None:
            context["packs"].clear()
            gc.collect()


if __name__ == "__main__":
    main()
