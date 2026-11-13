"""Fresh COCO/VOC Concept-SAE v2 selection, three-stage fitting and evaluation."""

import argparse
import gc
import importlib.util
from pathlib import Path

from historical_inputs import activate, audit_pack, own, read, separate, sha, write


def state_validator(code):
    spec = importlib.util.spec_from_file_location(
        "fresh_native_validation",
        code / "sources/geometry_consistency/native_sources.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_concept


def context(root, source, code, cohort, coco_root=None, voc_root=None):
    activate(code)
    from fresh_controls import context as control_context
    import concept_sae_v2 as revised

    if Path(revised.__file__).resolve() != code / "training/concept_sae_v2.py":
        raise ValueError(
            "Use the supplied development-repaired Concept-SAE v2 implementation"
        )
    separate(root, source, code)
    dependencies = (
        [source]
        + ([coco_root] if coco_root is not None else [])
        + ([voc_root] if voc_root is not None else [])
    )
    own(root, code, dependencies, __file__, "Concept-SAE-" + cohort)
    parent = control_context(root / "public_inputs", source, code, cohort, voc_root)
    selection = None
    if cohort == "voc":
        if coco_root is None:
            raise ValueError(
                "Provide --coco-concept-root with a completed fresh COCO tokenizer selection"
            )
        receipt = read(coco_root / "SELECTION.json")
        if receipt["protocol"] != revised.SPEC or not receipt["selection"].get(
            "success"
        ):
            raise ValueError(
                "Missing successful COCO development-only tokenizer selection"
            )
        if receipt["cohort"] != "coco":
            raise ValueError("Transfer the COCO selection to VOC")
        selection = receipt["selection"]
    ctx = revised.new_context(parent, cohort, selection)
    ctx["driver_identity"] = dict(
        driver_sha256=sha(__file__),
        cohort=cohort,
        parent_run=parent["store"].run_id,
        protocol=revised.SPEC,
    )
    marker = root / "FRESH_CONCEPT.json"
    if marker.exists() and read(marker) != ctx["driver_identity"]:
        raise ValueError("Concept-SAE driver or inputs changed")
    write(marker, ctx["driver_identity"])
    return ctx, selection


def train(root, ctx, cohort, selection):
    import concept_sae_v2 as revised

    if cohort == "coco":
        selection = revised.select_tokenizer_lr(ctx)
    write(
        root / "SELECTION.json",
        dict(
            cohort=cohort,
            selection=selection,
            protocol=revised.SPEC,
            driver_identity=ctx["driver_identity"],
        ),
    )
    models = []
    for seed in revised.SPEC["seeds"]:
        model, state = revised.fit(ctx, seed, selection)
        models.append(
            dict(
                seed=seed,
                key=f"concept_sae_final_s{seed}",
                weights_sha=state["weights_sha"],
                epochs=[r["epochs_completed"] for r in state["stage_receipts"]],
            )
        )
        del model, state
        gc.collect()
    if any(r["epochs"] != [30, 50, 30] for r in models):
        raise ValueError("Incomplete Concept-SAE stage budget")
    ctx["store"].save(
        "revised_concept_complete", dict(complete=True, seeds=[21, 22, 23])
    )
    write(
        root / "FITS_COMPLETE.json",
        dict(
            complete=True,
            models=models,
            store_root=str(ctx["store"].root),
            results=str(ctx["store"].results),
            categories=ctx["cfg"]["category_indices"],
            driver_identity=ctx["driver_identity"],
        ),
    )
    revised.archive(ctx, True)


def frozen(root, ctx, code):
    from coco_coordinated.run import required

    receipt = read(root / "FITS_COMPLETE.json")
    if (
        not receipt.get("complete")
        or receipt["driver_identity"] != ctx["driver_identity"]
    ):
        raise ValueError("Freeze all Concept-SAE fits before evaluation")
    if len(receipt["models"]) != 3 or {r["seed"] for r in receipt["models"]} != {
        21,
        22,
        23,
    }:
        raise ValueError("Keep all three Concept-SAE training orders")
    validate = state_validator(code)
    for row in receipt["models"]:
        state = required(ctx["store"], row["key"])
        validate(state, row["seed"])
        if state["weights_sha"] != row["weights_sha"]:
            raise ValueError("Concept-SAE endpoint changed")
    return receipt


def evaluate(root, ctx, code, cohort):
    import concept_sae_v2 as revised
    import concept_sae as native
    from coco_coordinated.run import required

    receipt = frozen(root, ctx, code)
    pack = (
        audit_pack(ctx, code, root, "controls", receipt)
        if cohort == "coco"
        else ctx["packs"]["eval"]
    )
    reports = []
    for row in receipt["models"]:
        meta = required(ctx["store"], row["key"])
        model = revised.make_model(ctx, row["seed"])
        model.load_state_dict(meta["weights"])
        result = native.evaluate(ctx, row["seed"], model, meta, pack, cohort)
        result["report"].update(
            training_protocol=meta["protocol"], stage_receipts=meta["stage_receipts"]
        )
        ctx["store"].save(f"concept_sae_{cohort}_s{row['seed']}", result)
        reports.append(result["report"])
        del model, meta
        gc.collect()
    write(
        root / "RESULTS_COMPLETE.json",
        dict(
            complete=True,
            cohort=cohort,
            reports=reports,
            store_root=str(ctx["store"].root),
            audit_identity=pack["identity"],
        ),
    )
    revised.archive(ctx, True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--features-root", required=True, type=Path)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--cohort", choices=["coco", "voc"], default="coco")
    p.add_argument("--coco-concept-root", type=Path)
    p.add_argument("--voc-inputs-root", type=Path)
    p.add_argument("--phase", choices=["train", "evaluate"], required=True)
    a = p.parse_args()
    root, source, code = (
        a.root.resolve(),
        a.features_root.resolve(),
        a.code_root.resolve(),
    )
    coco = None if a.coco_concept_root is None else a.coco_concept_root.resolve()
    ctx, selection = context(
        root,
        source,
        code,
        a.cohort,
        coco,
        a.voc_inputs_root.resolve() if a.voc_inputs_root else None,
    )
    train(root, ctx, a.cohort, selection) if a.phase == "train" else evaluate(
        root, ctx, code, a.cohort
    )


if __name__ == "__main__":
    main()
