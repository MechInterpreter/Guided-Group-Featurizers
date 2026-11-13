"""Plan or execute the public-input paper reproduction dependency graph."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from command_runner import run_command
from historical_inputs import own, read, write
from fresh_legacy import DEPENDENCIES


def plan(code, root, feature_cache=None):
    tasks = {}
    features = root / "matched75"
    voc = root / "voc19"

    def add(
        name, script, args, dependencies=(), receipt=None, python=None, annotated=False
    ):
        tasks[name] = dict(
            command=[
                str(python or sys.executable),
                str(code / "bootstrap" / (script + ".py")),
                "--code-root",
                str(code),
                *map(str, args),
            ],
            dependencies=list(dependencies),
            receipt=str(receipt) if receipt else None,
            annotated=annotated,
        )

    def matched(name, phase, dependencies, extra=(), receipt=None):
        add(
            name,
            "fresh_matched75",
            ["--root", features, "--phase", phase, *extra],
            dependencies,
            receipt,
        )

    matched("matched.prepare", "prepare", [], receipt=features / "FRESH_RUN.json")
    matched(
        "matched.extract",
        "extract",
        ["matched.prepare"],
        extra=[] if feature_cache is None else ["--feature-cache", feature_cache],
        receipt=features / "features/FEATURES.json",
    )
    matched(
        "matched.train_full",
        "train",
        ["matched.extract"],
        ["--regime", "full_category"],
        features / "training/full_category/BANKS.json",
    )
    matched(
        "matched.train_split",
        "train",
        ["matched.extract"],
        ["--regime", "category_split"],
        features / "training/category_split/BANKS.json",
    )
    matched(
        "matched.audit",
        "extract-audit",
        ["matched.train_full", "matched.train_split"],
        receipt=features / "features/FEATURES.json",
    )
    add(
        "matched.collect",
        "fresh_evaluation",
        ["--root", features, "--phase", "collect"],
        ["matched.audit"],
        features / "evaluation/COLLECTION_COMPLETE.json",
    )
    add(
        "matched.cards",
        "fresh_evaluation",
        ["--root", features, "--phase", "package"],
        ["matched.collect"],
        features / "evaluation/PACKAGING_COMPLETE.json",
    )
    for phase in ["lock-presence", "select-presence", "concentration"]:
        previous = (
            "matched.cards"
            if phase == "lock-presence"
            else "matched.lock-presence"
            if phase == "select-presence"
            else "matched.select-presence"
        )
        add(
            "matched." + phase,
            "fresh_evaluation",
            ["--root", features, "--phase", phase],
            [previous],
            annotated=True,
        )
    add(
        "matched.coherence",
        "fresh_reviews",
        ["--root", features, "--phase", "score"],
        ["matched.cards"],
        features / "evaluation/WHOLE_COHERENCE_COMPLETE.json",
        annotated=True,
    )

    add(
        "components.train",
        "fresh_components",
        [
            "--root",
            root / "components",
            "--features-root",
            features,
            "--phase",
            "train",
        ],
        ["matched.extract"],
        root / "components/FITS_COMPLETE.json",
    )
    add(
        "components.evaluate",
        "fresh_components",
        [
            "--root",
            root / "components",
            "--features-root",
            features,
            "--phase",
            "evaluate",
        ],
        ["components.train"],
        root / "components/RESULTS_COMPLETE.json",
    )
    for cohort in ["coco", "voc"]:
        folder = root / ("controls_" + cohort)
        args = ["--root", folder, "--features-root", features, "--cohort", cohort]
        if cohort == "voc":
            args += ["--voc-inputs-root", voc]
        for phase in ["train", "evaluate", "analysis"]:
            deps = (
                ["matched.extract"]
                if phase == "train"
                else ["controls." + cohort + ".train"]
            )
            if cohort == "voc":
                deps += ["voc19.prepare"]
            add(
                "controls." + cohort + "." + phase,
                "fresh_controls",
                [*args, "--phase", phase],
                deps,
                folder
                / (
                    {
                        "train": "FITS_COMPLETE.json",
                        "evaluate": "RESULTS_COMPLETE.json",
                        "analysis": "ANALYSIS_COMPLETE.json",
                    }[phase]
                ),
            )
    for phase in ["train", "evaluate"]:
        add(
            "supervised5." + phase,
            "fresh_supervised_five",
            [
                "--root",
                root / "supervised5",
                "--features-root",
                features,
                "--components-root",
                root / "components",
                "--phase",
                phase,
            ],
            ["components.train"] if phase == "train" else ["supervised5.train"],
            root
            / "supervised5"
            / ("FITS_COMPLETE.json" if phase == "train" else "RESULTS_COMPLETE.json"),
        )
    for phase in ["prepare", "train", "evaluate"]:
        add(
            "voc19." + phase,
            "fresh_voc",
            ["--root", voc, "--phase", phase],
            []
            if phase == "prepare"
            else ["voc19.prepare"]
            if phase == "train"
            else ["voc19.train"],
            voc
            / (
                {
                    "prepare": "INPUTS_COMPLETE.json",
                    "train": "FITS_COMPLETE.json",
                    "evaluate": "RESULTS.json",
                }[phase]
            ),
        )
    for cohort in ["coco", "voc"]:
        folder = root / ("concept_" + cohort)
        args = ["--root", folder, "--features-root", features, "--cohort", cohort]
        deps = ["matched.extract"]
        if cohort == "voc":
            args += [
                "--coco-concept-root",
                root / "concept_coco",
                "--voc-inputs-root",
                voc,
            ]
            deps += ["concept.coco.train", "voc19.prepare"]
        for phase in ["train", "evaluate"]:
            add(
                "concept." + cohort + "." + phase,
                "fresh_concept",
                [*args, "--phase", phase],
                deps if phase == "train" else ["concept." + cohort + ".train"],
                folder
                / (
                    "FITS_COMPLETE.json"
                    if phase == "train"
                    else "RESULTS_COMPLETE.json"
                ),
            )
    for phase in ["geometry", "controlled"]:
        folder = root / ("analysis_" + phase)
        args = [
            "--root",
            folder,
            "--features-root",
            features,
            "--controls-root",
            root / "controls_coco",
            "--supervised-root",
            root / "supervised5",
            "--concept-root",
            root / "concept_coco",
            "--phase",
            phase,
        ]
        add(
            "analysis." + phase,
            "fresh_analysis",
            args,
            ["controls.coco.evaluate", "supervised5.train", "concept.coco.train"],
            folder / (phase + "_COMPLETE.json"),
        )
    add(
        "compute",
        "fresh_compute",
        [
            "--root",
            root / "compute",
            "--features-root",
            features,
            "--voc19-root",
            voc,
            "--voc1024-root",
            root / "controls_voc",
            "--concept-root",
            root / "concept_voc",
        ],
        ["voc19.train", "controls.voc.train", "concept.voc.train"],
        root / "compute/COMPLETE.json",
    )

    baseline_python = (
        root
        / "environments/zoo_baselines"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    add(
        "zoo.generate",
        "fresh_zoo",
        ["--root", root / "zoo", "--phase", "generate"],
        [],
        root / "zoo/DATA.json",
        python=baseline_python,
    )
    for phase in ["train-baselines", "train-ggf", "audit-baselines", "audit-ggf"]:
        deps = (
            ["zoo.generate"]
            if phase.startswith("train")
            else ["zoo.train-baselines", "zoo.train-ggf"]
        )
        add(
            "zoo." + phase,
            "fresh_zoo",
            ["--root", root / "zoo", "--phase", phase],
            deps,
            root
            / "zoo"
            / (
                {
                    "train-baselines": "BASELINES_COMPLETE.json",
                    "train-ggf": "GUIDED_COMPLETE.json",
                    "audit-baselines": "BASELINE_RESULTS.json",
                    "audit-ggf": "GUIDED_RESULTS.json",
                }[phase]
            ),
            python=baseline_python if "baselines" in phase else None,
        )
    for stage, parents in DEPENDENCIES.items():
        args = [
            "--root",
            root / "legacy",
            "--features-root",
            features,
            "--stage",
            stage,
        ]
        add(
            "legacy." + stage + ".train",
            "fresh_legacy",
            [*args, "--phase", "train"],
            ["matched.extract", *["legacy." + name + ".train" for name in parents]],
            root / "legacy" / stage / "FITS_COMPLETE.json",
        )
        add(
            "legacy." + stage + ".evaluate",
            "fresh_legacy",
            [*args, "--phase", "evaluate"],
            ["legacy." + stage + ".train"],
            root / "legacy" / stage / "RESULTS_COMPLETE.json",
        )
    return tasks


def ordered(tasks, requested):
    result = []
    active = set()
    seen = set()

    def visit(name):
        if name not in tasks:
            raise ValueError("Unknown pipeline stage: " + name)
        if name in active:
            raise ValueError("Cyclic pipeline dependencies")
        if name in seen:
            return
        active.add(name)
        for parent in tasks[name]["dependencies"]:
            visit(parent)
        active.remove(name)
        seen.add(name)
        result.append(name)

    for name in requested:
        visit(name)
    return result


def prepare_baseline_environment(root, code):
    directory = root / "environments/zoo_baselines"
    python = directory / (
        "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    )
    if not python.exists():
        run_command(
            [sys.executable, "-m", "venv", "--system-site-packages", str(directory)],
            root / "logs",
            "zoo_environment",
        )
    run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "torch==2.11.0+cu128",
            "torchvision==0.26.0+cu128",
            "torchaudio==2.11.0+cu128",
            "--index-url",
            "https://download.pytorch.org/whl/cu128",
        ],
        root / "logs",
        "zoo_wheels",
    )
    versions = read(code / "evidence/zoo_guided/source/zoo_reference.json")[
        "original_versions"
    ]
    run_command(
        [
            str(python),
            "-m",
            "pip",
            "install",
            *[
                name + "==" + value
                for name, value in versions.items()
                if name != "torch"
            ],
        ],
        root / "logs",
        "zoo_numeric_versions",
    )


def run(root, code, tasks, names):
    own(root, code, [], __file__, "full-paper-reproduction")
    source = {
        p.relative_to(code).as_posix(): sha256(p.read_bytes()).hexdigest()
        for p in code.rglob("*.py")
    }
    source.update(
        {
            p.relative_to(code).as_posix(): sha256(p.read_bytes()).hexdigest()
            for p in (code / "bootstrap").glob("*.json")
        }
    )
    marker = root / "PIPELINE_SOURCE.json"
    if marker.exists() and read(marker) != source:
        raise ValueError("Pipeline source changed; choose a new reproduction directory")
    write(marker, source)
    completion = root / "PIPELINE_RUN_COMPLETE.json"
    if completion.is_file():
        completion.unlink()
    baseline_ready = False
    for name in names:
        task = tasks[name]
        receipt = root / "progress" / (name + ".json")
        if receipt.exists() and not task["annotated"]:
            saved = read(receipt)
            if (
                saved.get("complete")
                and saved["command"] == task["command"]
                and (not task["receipt"] or task["receipt"] in saved["outputs"])
                and all(
                    Path(path).is_file()
                    and receipt_digest(name, read(path)) == expected
                    for path, expected in saved["outputs"].items()
                )
            ):
                print("Completed stage:", name, flush=True)
                continue
            raise ValueError("Completed pipeline stage or its output changed: " + name)
        if "zoo_baselines" in task["command"][0] and not baseline_ready:
            prepare_baseline_environment(root, code)
            baseline_ready = True
        run_command(task["command"], root / "logs", name)
        outputs = {}
        if task["receipt"]:
            target = Path(task["receipt"])
            value = read(target)
            # Preparation receipts identify their input stores; fit/score receipts
            # additionally carry the worker's complete flag.
            if "complete" in value and not value["complete"]:
                raise ValueError("Stage did not complete: " + name)
            outputs[str(target)] = receipt_digest(name, value)
        write(receipt, dict(complete=True, command=task["command"], outputs=outputs))
    write(
        root / "PIPELINE_RUN_COMPLETE.json",
        dict(
            complete=True,
            stages=names,
            annotation_stages_included=any(tasks[name]["annotated"] for name in names),
            all_workflows_executed=all(name in names for name in tasks),
            paper_result_equivalence_verified=False,
            source_sha256=sha256(marker.read_bytes()).hexdigest(),
        ),
    )
    print(
        "Requested pipeline stages completed:",
        root / "PIPELINE_RUN_COMPLETE.json",
        flush=True,
    )


def receipt_digest(stage, value):
    """The audit phase extends the feature index without changing development inputs."""
    if stage == "matched.extract":
        value = dict(
            run=value["run"],
            normalization=value["normalization"],
            packs={
                name: value["packs"][name] for name in ["train", "select", "confirm"]
            },
        )
    elif stage == "matched.audit":
        value = dict(
            run=value["run"],
            normalization=value["normalization"],
            audit_training_receipts=value["audit_training_receipts"],
            audit=value["packs"]["audit"],
        )
    return sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    p.add_argument("--stages", nargs="+")
    p.add_argument("--include-annotations", action="store_true")
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    root, code = a.root.resolve(), a.code_root.resolve()
    tasks = plan(code, root, a.feature_cache)
    requested = a.stages or [
        name
        for name, task in tasks.items()
        if a.include_annotations or not task["annotated"]
    ]
    names = ordered(tasks, requested)
    if not a.execute:
        print(
            json.dumps(
                dict(
                    stages=[dict(name=name, **tasks[name]) for name in names],
                    execute=False,
                ),
                indent=2,
            )
        )
        return
    run(root, code, tasks, names)


if __name__ == "__main__":
    main()
