"""Fresh-process Colab entrypoint; sources are immutable and results save separately."""

import argparse, json, os, subprocess, sys, traceback, gc
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    o = json.loads(Path(p.parse_args().config).read_text())
    folder = Path(__file__).resolve().parent
    if o["stages"] not in [["all"], ["development"]]:
        raise ValueError("Unknown execution stages")
    from coco_ggf import matched_baselines as matched

    if matched.versions() != o["expected_versions"]:
        raise ValueError("Pinned numerical versions differ")
    import torch
    from threadpoolctl import threadpool_limits

    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime; T4 supported")
    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    import supervised_study as study
    from supervised_live import install_observers

    install_observers()
    ctx = None
    stage = "prepare"
    try:
        ctx = study.prepare(o)
        s = ctx["store"]
        for name in ["PROTOCOL.md", "README.md", "REFERENCE.md"]:
            (s.results / name).write_bytes((folder / name).read_bytes())
        s.write(
            "runtime.json",
            dict(
                gpu=torch.cuda.get_device_name(),
                versions=matched.versions(),
                stages=o["stages"],
            ),
        )
        Path(o["receipt"]).write_text(
            json.dumps(dict(complete=False, results=str(s.results), run_id=s.run_id))
        )
        stage = "annotations_and_independent_holdout_reservation"
        coco = study.annotations(ctx, o["source_output"])
        plan = study.plan_fresh(ctx, coco)
        # TRAIN-image replay verifies extraction math and authenticates before long fits.
        stage = "backbone_train_replay"
        source = (
            o["baseline_output"]
            if (
                Path(o["baseline_output"]) / "input_assets/dinov3/verified.json"
            ).exists()
            else o["source_output"]
        )
        study.backbone_preflight(ctx, coco, source)
        stage = "three_seed_supervised_development"
        choice = study.selection(ctx)
        partial = study.export(ctx, o["wrapper_zip"])
        if o["stages"] == ["development"]:
            Path(o["receipt"]).write_text(
                json.dumps(
                    dict(
                        complete=False,
                        development_complete=True,
                        results=str(s.results),
                        result_zip=str(partial),
                    )
                )
            )
            print("DEVELOPMENT COMPLETE; no test features opened:", partial, flush=True)
            return
        stage = "seven_seed_supervised_freeze"
        study.freeze(ctx, choice)
        study.export(ctx, o["wrapper_zip"])
        stage = "new_holdout_extraction"
        pack = study.extract_after_freeze(ctx, coco)
        stage = "supervised_holdout_evaluation_and_intervals"
        study.audit(ctx, pack)
        del pack
        gc.collect()
        matched.assert_sources(ctx)
        archive = study.export(ctx, o["wrapper_zip"], True)
        Path(o["receipt"]).write_text(
            json.dumps(
                dict(
                    complete=True,
                    results=str(s.results),
                    result_zip=str(archive),
                    run_id=s.run_id,
                    wrapper_sha=o["wrapper_sha"],
                    primary_seeds=7,
                    new_bsf_controls=0,
                    fresh_holdout_images=len(plan["splits"]["test"]),
                )
            )
        )
        print("SUPERVISED COMPARISON COMPLETE:", archive, flush=True)
    except BaseException as exc:
        if ctx is not None:
            try:
                study.flush(ctx)
            except BaseException as extra:
                print("Checkpoint flush failed:", repr(extra), flush=True)
            try:
                ctx["store"].write(
                    "last_interruption.json",
                    dict(
                        stage=stage,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    ),
                )
                partial = study.export(ctx, o["wrapper_zip"])
                Path(o["receipt"]).write_text(
                    json.dumps(
                        dict(
                            complete=False,
                            results=str(ctx["store"].results),
                            partial_zip=str(partial),
                            stage=stage,
                        )
                    )
                )
                print("Partial evidence ZIP:", partial, flush=True)
            except BaseException as extra:
                print("Partial export failed:", repr(extra), flush=True)
        raise
    finally:
        if ctx is not None:
            ctx["packs"].clear()
            gc.collect()


if __name__ == "__main__":
    main()
