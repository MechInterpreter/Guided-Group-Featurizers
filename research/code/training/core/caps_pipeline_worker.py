"""Fresh Python process: pinned packages require no notebook-kernel restart."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    options = json.loads(Path(args.config).read_text())
    root = Path(__file__).resolve().parent
    env = dict(
        os.environ,
        PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    print(
        "Running bundled CPU protocol, inference, resume and ordering checks...",
        flush=True,
    )
    tests = [
        "test_caps_pipeline.py",
        "test_cap_followup.py",
        "test_width_balance.py",
        "test_weight_bound.py",
        "test_weight_bound_transport.py",
    ]
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            *[str(root / "tests" / name) for name in tests],
            "--disable-warnings",
            "--basetemp",
            str(Path(options["local"]) / ("selftest_" + os.urandom(6).hex())),
        ],
        env=env,
        cwd=str(root),
    )
    import torch
    from threadpoolctl import threadpool_limits
    from coco_ggf import caps_pipeline as experiment

    if experiment.matched.versions() != options["expected_versions"]:
        raise RuntimeError(
            "Fresh worker numerical versions do not match the saved source"
        )
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    context = None
    stage = "prepare"
    try:
        context = experiment.prepare(
            options["paths"],
            options["training_output"],
            options["output"],
            options["local"],
            options["cache_local"],
            options["source_sha"],
        )
        context["store"].results.joinpath("PROTOCOL.md").write_bytes(
            (root / "PROTOCOL_COMBINED.md").read_bytes()
        )
        print(
            "Development progress:",
            context["training"]["store"].results / "progress.json",
            flush=True,
        )
        print(
            "Pipeline progress:", context["store"].results / "progress.json", flush=True
        )
        stage = "preflight"
        timing = experiment.preflight(context)
        print(
            "Estimated remaining gradient hours:",
            round(timing["estimated_gradient_hours"], 2),
            "(development scoring, Drive and final test scoring add time)",
            flush=True,
        )
        stage = "development_and_freeze"
        experiment.train_and_freeze(context)
        # Release training-only maps/masks before the full-size test audit.
        context["training"]["packs"].pop("train", None)
        stage = "prepare_evaluation"
        evaluation = experiment.prepare_evaluation(
            context, options.get("previous_test_output")
        )
        stage = "cached_test_loading"
        experiment.load_test(evaluation)
        stage = "test_evaluation"
        summary = experiment.evaluate(evaluation)
        stage = "complete"
        experiment.complete(context, summary)
        stage = "export"
        result = experiment.export(context, options["source_zip"])
        receipt = dict(
            complete=True,
            result_zip=str(result),
            pipeline_results=str(context["store"].results),
            development_results=str(context["training"]["store"].results),
            test_results=str(evaluation["store"].results),
            selected_development_cap=summary["development_selection"]["selected_cap"],
        )
        Path(options["receipt"]).write_text(json.dumps(receipt, indent=2))
        print("ALL STAGES COMPLETE. Results ZIP:", result, flush=True)
    except BaseException as exc:
        if context is not None:
            experiment.failure(context, stage, exc, options["source_zip"])
        raise


if __name__ == "__main__":
    main()
