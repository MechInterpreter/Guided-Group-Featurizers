"""Plan or explicitly execute the paper's reproducibility entry points."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "workflow_registry.json").read_text(encoding="utf-8"))
CORE_SHA = "dfed82ac09e358dd8fc5e8d6de1bb524519cc0965cfea45f1318367b7eab90cf"


def check_environment(args):
    """Check the original numerical environment without installing anything."""
    parent = "2921b057f824362ab83e76e7fb35645f94c376fadf8b86f4f22d1a2311105b9e"
    manifests = []
    for path in (args.input_root / "AGF_COCO_capacity_sweep_v1/results").glob(
        "*/manifest.json"
    ):
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("run_id") == parent:
            manifests.append(value)
    if len(manifests) != 1:
        raise ValueError(
            "Expected exactly one original capacity manifest with the recorded run identity"
        )
    manifest = manifests[0]
    digest = hashlib.sha256(
        json.dumps(manifest["config"], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    if digest != parent:
        raise ValueError("Original capacity manifest identity differs")
    rows = []
    for name in ("numpy", "torch", "scipy", "scikit-learn", "einops"):
        expected = manifest["config"]["versions"][name]
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        rows.append(
            dict(
                package=name,
                expected=expected,
                installed=installed,
                matches=installed == expected,
            )
        )
    print(json.dumps(rows, indent=2))
    if not all(row["matches"] for row in rows):
        raise ValueError(
            "Install the recorded numerical versions before executing the matched experiments"
        )


def read_config(path, keys):
    if path is None:
        raise ValueError(
            "This workflow requires --config with a recorded study configuration."
        )
    path = path.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    missing = set(keys) - value.keys()
    if missing:
        raise ValueError("Configuration is missing " + ", ".join(sorted(missing)))
    if "regime" in value and value["regime"] not in {"full_category", "category_split"}:
        raise ValueError("Unknown matched-study regime")
    return path, value


def plan(args):
    task = REGISTRY[args.workflow]
    source = ROOT / task["source"]
    program = source / task["entrypoint"]
    preparation = []
    if task.get("frozen_profile"):
        materializer = ROOT / "tools/materialize_frozen.py"
        profiles = json.loads(
            (ROOT / "frozen_sources/profiles.json").read_text(encoding="utf-8")
        )["profiles"]
        profile = profiles[task["frozen_profile"]]
        if task["entrypoint"] not in profile["files"]:
            raise ValueError("Entrypoint is not part of the frozen source profile")
        if not materializer.is_file():
            raise FileNotFoundError(materializer)
        preparation.append(
            [
                sys.executable,
                str(materializer),
                task["frozen_profile"],
                "--output",
                str(source),
            ]
        )
    elif not program.is_file():
        raise FileNotFoundError(program)
    command = [sys.executable, str(program)]
    if task["arguments"] == "config":
        config, values = read_config(args.config, task.get("required_config_keys", []))
        if (
            task.get("experiment_sha")
            and values["experiment_sha"] != task["experiment_sha"]
        ):
            raise ValueError("Configuration uses a different frozen source identity")
        if (
            "core_sha" in task.get("required_config_keys", [])
            and values["core_sha"] != CORE_SHA
        ):
            raise ValueError("Configuration uses a different frozen core identity")
        for key, expected in task.get("config_values", {}).items():
            if values.get(key) != expected:
                raise ValueError(f"This workflow requires {key}={expected!r}")
        if "paths" in task.get("required_config_keys", []):
            if (
                not isinstance(values["paths"], dict)
                or not {"capacity", "original", "strong"} <= values["paths"].keys()
            ):
                raise ValueError(
                    "Matched training requires capacity, original and strong input stores"
                )
        command += ["--config", str(config)]
    elif task["arguments"] == "root":
        if args.root is None:
            raise ValueError(
                "This workflow requires --root with its evaluation directory"
            )
        command += ["--root", str(args.root.expanduser().resolve())]
        if task.get("phase"):
            command += ["--phase", task["phase"]]
    command += task.get("extra_args", [])
    if task.get("input_root_argument"):
        if args.input_root is None:
            raise ValueError(
                "This workflow requires --input-root with the original input stores"
            )
        command += [
            task["input_root_argument"],
            str(args.input_root.expanduser().resolve()),
        ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(source), str(source / "core")])
    environment.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"
    )
    return command, source, environment, preparation


def configs(args):
    # Names match the documented upstream stores. The user supplies their common root.
    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    local_root = args.local_root.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    paths = {
        key: str(input_root / name)
        for key, name in {
            "capacity": "AGF_COCO_capacity_sweep_v1",
            "original": "AGF_COCO_stratified_confirmation_v1",
            "strong": "AGF_COCO_strong_mask_regroup_v1",
        }.items()
    }
    for label, write_root in (
        ("output", output_root),
        ("scratch", local_root),
        ("configuration", destination),
    ):
        for store in map(Path, paths.values()):
            store = store.resolve()
            if (
                write_root == store
                or write_root.is_relative_to(store)
                or store.is_relative_to(write_root)
            ):
                raise ValueError(
                    f"Choose a {label} root separate from the read-only input stores"
                )
    configurations = {}
    for group, source, training_sha, evaluation_sha in [
        (
            "21_23",
            "sources/matched75_original",
            "2502868f0b4e9553cb8d9cb0a1aac89f294e36000508ef98d5724aadcfe7d30c",
            "9aecc52a2102b67b7f4f6ed839f3fa1a946a7676b0ce59987dd9b0908ce67446",
        ),
        (
            "24_27",
            "training",
            "fc6f3f8579f9a953e5a1630b3baf6cacd7c3e1ae72f1e4858ee497201a926c98",
            "fc6f3f8579f9a953e5a1630b3baf6cacd7c3e1ae72f1e4858ee497201a926c98",
        ),
    ]:
        base = dict(
            paths=paths,
            drive_root=str(input_root),
            cache_local=str(local_root / "feature_cache"),
            core_sha=CORE_SHA,
            concept_source=str(ROOT / source / "concept_sae_native.py"),
            annotation_path="",
        )
        training_root = output_root / group / "training"
        for regime in ("full_category", "category_split"):
            configurations[f"{group}_{regime}.json"] = dict(
                base,
                regime=regime,
                output=str(training_root / regime),
                local=str(local_root / group / regime),
                experiment_sha=training_sha,
            )
        configurations[f"{group}_evaluation.json"] = dict(
            base,
            output=str(output_root / group / "evaluation"),
            local=str(local_root / group / "evaluation"),
            training_root=str(training_root),
            experiment_sha=evaluation_sha,
            seed21_pilot=False,
            response_chunk=16,
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in configurations.items():
        path = destination / name
        if path.exists():
            raise FileExistsError(
                f"Keep or explicitly remove the existing configuration first: {path}"
            )
    for name, content in configurations.items():
        (destination / name).write_text(
            json.dumps(content, indent=2) + "\n", encoding="utf-8"
        )
    print(
        f"Wrote {len(configurations)} configurations. Inspect local paths and input availability before executing."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    environment = actions.add_parser(
        "check-environment",
        help="Check the matched-study numerical versions against its parent manifest",
    )
    environment.add_argument("--input-root", type=Path, required=True)
    configuration = actions.add_parser("make-configs")
    for name in ("input-root", "output-root", "local-root", "destination"):
        configuration.add_argument("--" + name, type=Path, required=True)
    run = actions.add_parser("run")
    run.add_argument("workflow", choices=sorted(REGISTRY))
    run.add_argument("--config", type=Path)
    run.add_argument("--root", type=Path)
    run.add_argument("--input-root", type=Path)
    run.add_argument(
        "--execute",
        action="store_true",
        help="Execute the validated command. Default prints a plan.",
    )
    args = parser.parse_args()
    if args.action == "list":
        for name, task in REGISTRY.items():
            print(name + "  " + task["description"])
    elif args.action == "make-configs":
        configs(args)
    elif args.action == "check-environment":
        check_environment(args)
    else:
        command, cwd, environment, preparation = plan(args)
        print(
            json.dumps(
                {
                    "preparation": preparation,
                    "command": command,
                    "cwd": str(cwd),
                    "execute": args.execute,
                },
                indent=2,
            )
        )
        if args.execute:
            for prepare_command in preparation:
                subprocess.run(prepare_command, cwd=ROOT, check=True)
            subprocess.run(command, cwd=cwd, env=environment, check=True)


if __name__ == "__main__":
    main()
