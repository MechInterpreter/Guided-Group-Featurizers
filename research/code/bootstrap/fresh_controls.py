"""Fit the recorded 1,024-group comparisons from public-data feature stores."""

import argparse
import copy
from pathlib import Path

from historical_inputs import (
    activate,
    audit_ids,
    audit_pack,
    coco_context,
    numerical_environment,
    own,
    read,
    sha,
    separate,
    write,
)


def context(root, source, code, cohort="coco", voc_root=None):
    activate(code)
    import experiment
    from coco_ggf.width_balance import settings as balance_settings
    from scale_plan import settings

    versions = numerical_environment(code)
    if cohort == "coco":
        ctx = coco_context(
            root,
            source,
            code,
            "voc_slics_compute",
            __file__,
            dict(experiment.recipe(), cohort=cohort, audit_ids=audit_ids(code)),
        )
    elif cohort == "voc":
        from fresh_voc import context as voc_context

        own(
            root, code, [source] + ([voc_root] if voc_root else []), __file__, "VOC1024"
        )
        # Preparation creates VOC inputs using the original public-data loader.
        voc_root = voc_root or root / "public_voc"
        parent = voc_context(voc_root, code)
        payload = dict(
            schema="ggf-fresh-voc1024-v1",
            protocol=experiment.recipe(),
            versions=versions,
            driver_sha256=sha(__file__),
            inputs=read(voc_root / "INPUTS_COMPLETE.json"),
        )
        marker = root / "FRESH_VOC1024.json"
        if marker.exists() and read(marker) != payload:
            raise ValueError("VOC1024 source or input identity changed")
        from coco_adaptation.inputs import JournalStore
        from coco_ggf.drive_recovery import install_drive_recovery

        local = root / "scratch"
        local.mkdir(parents=True, exist_ok=True)
        store = JournalStore(root / "models", payload, local / "rescue", False)
        install_drive_recovery(store)
        ctx = dict(parent, store=store, local=local)
        ctx["cfg"].update(fresh_test_seed=126101)
        write(root / "FRESH_VOC1024.json", payload)
    else:
        raise ValueError("Choose coco or voc")
    ctx.update(
        rule=settings(),
        balance_recipe=copy.deepcopy(balance_settings()["balance"]),
        device="cuda",
        stamps=[],
        checkpoint_stamps=[],
        sources={},
        _pairing_proofs={},
        active_key=None,
    )
    ctx["options"] = dict(
        output=str(root / "models"),
        local=str(ctx["local"]),
        wrapper_sha=sha(__file__),
        source_sha=sha(__file__),
        source_output=str(root / "public_inputs/parent"),
        concept_source=str(code / "training/concept_sae_native.py"),
    )
    ctx["store"].save(
        "voc_plan", dict(category_indices=ctx["cfg"]["category_indices"])
    ) if cohort == "voc" else None
    return ctx


def validate_catalog(rows, kinds):
    expected = {(seed, kind) for seed in [21, 22, 23] for kind in kinds}
    if len(rows) != len(expected) or {(r["seed"], r["kind"]) for r in rows} != expected:
        raise ValueError("Incomplete or duplicate control endpoint catalog")
    import experiment

    for row in rows:
        experiment.validate_row(row)


def train(ctx, extras=True):
    import experiment

    rows = []
    for seed in experiment.SEEDS:
        rows.extend(experiment.endpoint(ctx, seed, kind) for kind in experiment.PRIMARY)
    validate_catalog(rows, experiment.PRIMARY)
    experiment.freeze(ctx, rows, "primary_frozen")
    if extras:
        for kind in experiment.EXTRA:
            rows.extend(
                experiment.endpoint(ctx, seed, kind) for seed in experiment.SEEDS
            )
    validate_catalog(rows, experiment.PRIMARY + (experiment.EXTRA if extras else []))
    experiment.freeze(ctx, rows, "all_models_frozen")


def frozen(ctx, extras=True):
    import experiment
    from coco_coordinated.run import required

    catalog = required(ctx["store"], "all_models_frozen")
    if not catalog.get("complete"):
        raise ValueError("Complete every declared fit before evaluation")
    validate_catalog(
        catalog["models"], experiment.PRIMARY + (experiment.EXTRA if extras else [])
    )
    experiment.check_pairing(ctx, catalog["models"])
    return catalog


def evaluate(ctx, root, code, cohort):
    import experiment, evaluation

    complete = frozen(ctx, cohort == "coco")
    pack = (
        audit_pack(ctx, code, root, "controls", complete)
        if cohort == "coco"
        else ctx["packs"]["eval"]
    )
    evaluation.score(ctx, complete["models"], pack, cohort)
    primary = [r for r in complete["models"] if r["kind"] in experiment.PRIMARY]
    evaluation.intervals(ctx, primary, cohort)
    for kind in experiment.EXTRA if cohort == "coco" else []:
        evaluation.intervals(
            ctx, [r for r in complete["models"] if r["kind"] == kind], cohort
        )
    ctx["store"].save("primary_complete", dict(complete=True, models=primary))
    write(
        root / "RESULTS_COMPLETE.json",
        dict(
            complete=True,
            cohort=cohort,
            models=complete["models"],
            results=str(ctx["store"].results),
            audit_identity=pack["identity"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--voc-inputs-root", type=Path)
    parser.add_argument("--cohort", choices=["coco", "voc"], default="coco")
    parser.add_argument(
        "--phase", choices=["train", "evaluate", "analysis"], required=True
    )
    args = parser.parse_args()
    root, code, source = (
        args.root.resolve(),
        args.code_root.resolve(),
        args.features_root.resolve(),
    )
    separate(root, code, source)
    ctx = context(
        root,
        source,
        code,
        args.cohort,
        args.voc_inputs_root.resolve() if args.voc_inputs_root else None,
    )
    if args.phase == "train":
        train(ctx, args.cohort == "coco")
        write(
            root / "FITS_COMPLETE.json",
            dict(
                complete=True,
                models=frozen(ctx, args.cohort == "coco")["models"],
                store_root=str(ctx["store"].root),
            ),
        )
    elif args.phase == "evaluate":
        evaluate(ctx, root, code, args.cohort)
    else:
        import analysis

        complete = frozen(ctx, args.cohort == "coco")
        pack = (
            audit_pack(ctx, code, root, "controls", complete)
            if args.cohort == "coco"
            else ctx["packs"]["eval"]
        )
        coco = None
        if args.cohort == "coco":
            from backbone import annotations

            coco = annotations(ctx, ctx["options"]["source_output"])
        analysis.run(ctx, complete["models"], pack, args.cohort, coco)
        write(
            root / "ANALYSIS_COMPLETE.json",
            dict(
                complete=True,
                models=complete["models"],
                results=str(ctx["store"].results),
            ),
        )


if __name__ == "__main__":
    main()
