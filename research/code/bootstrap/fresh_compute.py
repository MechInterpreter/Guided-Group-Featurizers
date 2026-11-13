"""Recreate the 17 execution profiles from freshly fitted VOC checkpoints."""

import argparse
from pathlib import Path

from historical_inputs import activate, own, read, write


def run(root, code, voc19_root, voc1024_root, concept_root, features_root):
    own(
        root,
        code,
        [voc19_root, voc1024_root, concept_root, features_root],
        __file__,
        "execution-profiles",
    )
    activate(code)
    import torch
    from fresh_voc import context as voc_context
    from fresh_controls import context as control_context, frozen
    from coco_baselines import training
    from coco_coordinated.run import FrozenStore, required
    from coco_baselines.storage import identity
    from compute_probe import benchmark, bsf_factory, native_factory, concept_factory

    ctx = voc_context(voc19_root, code)
    ctx["options"] = dict(
        ctx["options"], concept_source=str(code / "training/concept_sae_native.py")
    )
    receipt = read(voc19_root / "FITS_COMPLETE.json")
    if not receipt.get("complete"):
        raise ValueError("Complete all six fresh VOC19 fits first")
    from fresh_voc import validate_rows

    validate_rows(receipt["models"])
    primary = control_context(voc1024_root, features_root, code, "voc", voc19_root)
    catalog = frozen(primary, False)
    if any(
        primary["packs"][k]["ids"] != ctx["packs"][k]["ids"]
        for k in ["train", "select", "eval"]
    ) or identity(primary["norm"]) != identity(ctx["norm"]):
        raise ValueError("VOC profile inputs or normalization differ")
    concept = read(concept_root / "FITS_COMPLETE.json")
    if not concept.get("complete"):
        raise ValueError("Complete all three fresh VOC Concept-SAE fits first")
    source = FrozenStore(Path(concept["store_root"]), root / "read_concept")
    if (
        source.config["cohort"] != "voc"
        or source.config["train"] != ctx["packs"]["train"]["identity"]
        or source.config["normalization"] != identity(ctx["norm"])
    ):
        raise ValueError("Concept-SAE uses another VOC feature cohort")
    root.mkdir(parents=True, exist_ok=True)
    profiles = []
    ids = list(range(8))

    def profile(name, factory, images, **facts):
        value = dict(name=name, **facts, **benchmark(factory, images))
        profiles.append(value)
        write(root / "compute_profiles.json", profiles)

    for row in [r for r in receipt["models"] if r["seed"] == 21]:
        st = required(ctx["store"], row["key"])
        weights = {"D": st["D"]} if row["arm"] == "SLiCS" else st["weights"]
        if training.weight_sha(weights) != row["weights_sha"]:
            raise ValueError("VOC19 weights changed")
        if row["arm"] == "GGF":
            for phase, learn, adapt in [
                ("inference", False, False),
                ("training_plain", True, False),
                ("training_controller", True, True),
            ]:
                profile(
                    "GGF19_" + phase,
                    bsf_factory(ctx, row, st, ids, learn, adapt),
                    8,
                    method="GGF19",
                    phase=phase,
                    learned_parameters=row["learned_parameters"],
                )
        else:
            for phase in ["partial", "joint", "training"]:
                profile(
                    "SLiCS19_" + phase,
                    native_factory(ctx, st["D"], ids, phase),
                    8,
                    method="SLiCS19",
                    phase=phase,
                    learned_parameters=row["learned_parameters"],
                )
    for row in [r for r in catalog["models"] if r["seed"] == 21]:
        st = required(primary["store"], row["key"])
        if training.weight_sha(st["weights"]) != row["weights_sha"]:
            raise ValueError("VOC1024 weights changed")
        for phase, learn, adapt in [
            ("inference", False, False),
            ("training_plain", True, False),
        ] + ([("training_controller", True, True)] if row["kind"] == "agf_g" else []):
            profile(
                row["kind"] + "1024_" + phase,
                bsf_factory(ctx, row, st, ids, learn, adapt),
                8,
                method=row["kind"] + "1024",
                phase=phase,
                learned_parameters=row["total_learned_parameters"],
            )
    st = required(source, "concept_sae_final_s21")
    if training.weight_sha(st["weights"]) != st["weights_sha"] or [
        r["epochs_completed"] for r in st["stage_receipts"]
    ] != [30, 50, 30]:
        raise ValueError("Incomplete or changed Concept-SAE v2 endpoint")
    profile(
        "ConceptSAE_inference",
        concept_factory(ctx, st, ids),
        8,
        method="Concept-SAE",
        phase="inference",
        learned_parameters=st["parameters"],
    )
    for stage in [1, 2, 3]:
        profile(
            "ConceptSAE_training_stage" + str(stage),
            concept_factory(ctx, st, list(range(64)), stage),
            64,
            method="Concept-SAE",
            phase="training_stage" + str(stage),
            learned_parameters=st["parameters"],
        )
    if len(profiles) != 17:
        raise ValueError("Execution-profile inventory is incomplete")
    write(
        root / "COMPLETE.json",
        dict(
            complete=True,
            profiles=17,
            GPU=torch.cuda.get_device_name(),
            full_FLOPs_available=False,
            scope="Original three-repeat disposable-model benchmark and registered operator coverage",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "features-root", "voc19-root", "voc1024-root", "concept-root"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    a = p.parse_args()
    run(
        a.root.resolve(),
        a.code_root.resolve(),
        a.voc19_root.resolve(),
        a.voc1024_root.resolve(),
        a.concept_root.resolve(),
        a.features_root.resolve(),
    )


if __name__ == "__main__":
    main()
