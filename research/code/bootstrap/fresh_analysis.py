"""Run frozen-model geometry and transformations on fresh study outputs."""

import argparse
import gc
from pathlib import Path

from historical_inputs import (
    activate,
    open_study,
    read,
    sha,
    write,
    separate,
    InputView,
)


def inputs(root, code, features, controls, supervised, concept, profile):
    separate(root, code, features, controls, supervised, concept)
    owner = dict(
        driver_sha256=sha(__file__),
        profile=profile,
        controls=str(controls),
        supervised=str(supervised),
        concept=str(concept),
    )
    marker = root / "ANALYSIS_OWNER.json"
    if root.exists() and any(root.iterdir()) and not marker.exists():
        raise ValueError("Use an empty analysis directory")
    if marker.exists() and read(marker) != owner:
        raise ValueError("Analysis source or dependencies changed")
    root.mkdir(parents=True, exist_ok=True)
    write(marker, owner)
    activate(code, profile)
    # Bind the requested analysis edition before the shared control context imports.
    from fresh_controls import context as control_context, frozen
    from coco_coordinated.run import FrozenStore, required
    from coco_baselines.storage import identity
    from coco_baselines import training
    import native_sources

    ctx = control_context(controls, features, code, "coco")
    # The shared context loads the control edition; restore the requested
    # analysis edition before importing its study-specific modules.
    activate(code, profile)
    complete = frozen(ctx, True)
    primary = [
        r
        for r in complete["models"]
        if r["kind"] in ["agf_g", "supervised_vanilla", "released_vanilla"]
    ]
    source = FrozenStore(ctx["store"].root, root / "read_primary")
    ctx["local"] = root / "scratch"
    ctx["local"].mkdir(exist_ok=True)
    for row in primary:
        st = required(source, row["key"])
        if training.weight_sha(st["weights"]) != row["weights_sha"]:
            raise ValueError("Fresh BSF weights changed")
        import torch

        folder = ctx["local"] / "frozen_weights"
        folder.mkdir(exist_ok=True)
        torch.save(st["weights"], folder / (row["weights_sha"] + ".pt"))
    native = []
    stores = {}
    native_source = open_study(supervised, root / "read_slics")
    receipt = read(supervised / "FITS_COMPLETE.json")
    if not receipt.get("complete"):
        raise ValueError("Finish the fresh five-order fits first")
    for seed in [21, 22, 23]:
        row = next(
            r
            for r in receipt["models"]
            if r["seed"] == seed and r["arm"] == "slics_final_partial"
        )
        st = required(native_source, row["key"])
        if (
            st["updates"] != 10640
            or training.weight_sha({"D": st["D"]}) != row["weights_sha"]
        ):
            raise ValueError("SLiCS endpoint changed")
        native.append(
            dict(
                kind="slics",
                seed=seed,
                key=row["key"],
                weights_sha=row["weights_sha"],
                slots={
                    int(t["category_index"]): int(t["contrast_group"])
                    for t in row["targets"]
                },
                source_run=native_source.run_id,
            )
        )
    stores["slics"] = native_source
    receipt = read(concept / "FITS_COMPLETE.json")
    if not receipt.get("complete"):
        raise ValueError("Finish the fresh COCO Concept-SAE fits first")
    store = FrozenStore(Path(receipt["store_root"]), root / "read_concept")
    if (
        store.config["cohort"] != "coco"
        or store.config["train"] != ctx["packs"]["train"]["identity"]
        or store.config["normalization"] != identity(ctx["norm"])
    ):
        raise ValueError("Concept-SAE source cohort differs")
    for seed in [21, 22, 23]:
        st = required(store, f"concept_sae_final_s{seed}")
        native_sources.validate_concept(st, seed)
        native.append(
            dict(
                kind="concept_sae",
                seed=seed,
                key=f"concept_sae_final_s{seed}",
                weights_sha=st["weights_sha"],
                slots={int(c): j for j, c in enumerate(store.config["categories"])},
                source_run=store.run_id,
            )
        )
    stores["concept_sae"] = store
    for s in stores.values():
        actual = s.config.get("training_identity", s.config.get("train"))
        if actual != ctx["packs"]["train"]["identity"]:
            raise ValueError("Native training feature cohort differs")
    options = dict(
        ctx["options"],
        output=str(root),
        local=str(root / "scratch"),
        receipt=str(root / "COMPLETE.json"),
        analysis_sha=sha(__file__),
        source_output=str(root / "public_parent"),
        concept_source=str(code / "training/concept_sae_native.py"),
    )
    ctx.update(
        options=options,
        cats=list(ctx["cfg"]["category_indices"]),
        normalization=identity(ctx["norm"]),
    )
    protocol = dict(
        schema="fresh-frozen-analysis-v1",
        profile=profile,
        driver_sha256=sha(__file__),
        models=[
            dict(kind=r["kind"], seed=r["seed"], weights_sha=r["weights_sha"])
            for r in primary + native
        ],
        calibration=ctx["packs"]["select"]["identity"],
        evaluation=ctx["packs"]["confirm"]["identity"],
        normalization=identity(ctx["norm"]),
    )
    from coco_adaptation.inputs import JournalStore
    from coco_ggf.drive_recovery import install_drive_recovery

    job = JournalStore(root / "analysis", protocol, ctx["local"] / "rescue", False)
    install_drive_recovery(job)
    ctx["store"] = InputView(
        job, {key: source for key in ["primary_frozen", "primary_complete"]}
    )
    ctx["signature"] = identity(protocol)
    ctx["out"] = root / "results" / ctx["signature"]
    ctx["out"].mkdir(parents=True, exist_ok=True)
    for d in [
        "guided_geometry",
        "dictionary_geometry",
        "discovery",
        "controlled_consistency",
        "response_scales",
    ]:
        (ctx["out"] / d).mkdir(exist_ok=True)
    write(ctx["out"] / "FRESH_PROTOCOL.json", protocol)
    return ctx, source, stores, primary, native


def geometry(ctx, source, stores, rows):
    from study_math import panel
    from bank_inspection import run_stages

    run_stages(
        ctx,
        rows,
        source,
        stores,
        panel(ctx["packs"]["select"], ctx["cats"]),
        panel(ctx["packs"]["confirm"], ctx["cats"]),
    )


def controlled(ctx, source, stores, rows, native_rows, root, code):
    import torch
    import cv_study as cv, analysis, backbone, native_coherence, native_sources
    from cv_math import Cache, digest
    from native_units import BSFUnits
    from historical_inputs import audit_pack

    frozen = dict(complete=True, models=rows + native_rows)
    pack = audit_pack(ctx, code, root, "controls", frozen)
    coco = backbone.annotations(ctx, ctx["options"]["source_output"])
    backbone.backbone_preflight(ctx, coco, ctx["options"]["source_output"])
    records = cv.select_cohort(ctx["packs"]["train"], ctx, "train") + cv.select_cohort(
        pack, ctx, "eval"
    )
    records = cv.geometry_preflight(ctx, coco, records)
    ctx["cohort"] = records
    feature_id = digest(
        dict(
            protocol=ctx["signature"],
            cohort=records,
            backbone=ctx["provenance"]["backbone_weights"],
        )
    )
    features = Cache(
        root / "scratch/features" / feature_id,
        root / "features" / feature_id,
        feature_id,
    )
    cv.extract(
        ctx, coco, records, features, Path(ctx["fresh_provenance"]["backbone"]["model"])
    )
    ctx["mask_probe"] = analysis.fit_probe(ctx)
    outcomes = []
    semantic = []
    categories = sorted({r["category"] for r in records})
    for row in rows:
        model = analysis.load_model(ctx, row)
        plans = cv.get_plan(ctx, row, model, source)
        signature = digest(
            dict(features=feature_id, weights=row["weights_sha"], plans=plans)
        )
        cache = Cache(
            root / "scratch/model_analysis" / signature,
            root / "model_analysis" / signature,
            signature,
        )
        summaries, pcas, D = cv.collect(
            ctx, row, model, records, features, cache, plans
        )
        outcomes.append(cv.evaluate_model(ctx, row, summaries, pcas, D, plans))
        common, observations = native_coherence.bsfunits(
            row, summaries, D, plans, categories
        )
        adapter = BSFUnits(model, common["units"], ctx["device"])
        semantic.append(
            native_coherence.process(
                ctx, common, adapter, observations, categories, features, coco
            )
        )
        if row["seed"] == 21:
            cv.atlas(ctx, row, model, summaries, pcas, D, plans, features, coco)
        del model, adapter, observations, summaries
        gc.collect()
        torch.cuda.empty_cache()
    cv.finish(
        ctx,
        outcomes,
        records,
        rows,
        dict(revision="fresh-inputs", transformations=cv.LEVELS),
    )
    for row in native_rows:
        adapter = native_sources.load(ctx, row, stores[row["kind"]], categories)
        common = dict(
            row,
            units=[row["slots"][c] for c in categories],
            interface=adapter.interface,
        )
        observations = native_coherence.collect(ctx, common, adapter, records, features)
        semantic.append(
            native_coherence.process(
                ctx, common, adapter, observations, categories, features, coco
            )
        )
        del adapter, observations
        gc.collect()
        torch.cuda.empty_cache()
    native_coherence.finish(ctx, semantic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "root",
        "features-root",
        "controls-root",
        "supervised-root",
        "concept-root",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--phase", choices=["geometry", "controlled"], required=True)
    a = p.parse_args()
    root, code = a.root.resolve(), a.code_root.resolve()
    profile = (
        "geometry_consistency" if a.phase == "geometry" else "controlled_variation"
    )
    ctx, source, stores, primary, native = inputs(
        root,
        code,
        a.features_root.resolve(),
        a.controls_root.resolve(),
        a.supervised_root.resolve(),
        a.concept_root.resolve(),
        profile,
    )
    if a.phase == "geometry":
        geometry(ctx, source, stores, primary + native)
    elif a.phase == "controlled":
        controlled(ctx, source, stores, primary, native, root, code)
    write(
        root / (a.phase + "_COMPLETE.json"),
        dict(complete=True, results=str(ctx["out"]), models=15),
    )


if __name__ == "__main__":
    main()
