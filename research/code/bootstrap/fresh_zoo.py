"""Train and evaluate the synthetic Manifold Zoo comparison from generated data."""

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def modules(code):
    source = code / "evidence/zoo_guided/source"
    core = code / "training/core"
    sys.path[:0] = [str(source), str(core)]
    import focused_zoo
    from manifold_zoo import run as native
    from zoo_benchmark import data

    return focused_zoo, native, data


def specification(code):
    reference = read(code / "evidence/zoo_guided/source/zoo_reference.json")
    recorded = read(code / "evidence/zoo_guided/results/manifest.json")["config"]
    if reference["config"] != recorded["recipe"]["config"]:
        raise ValueError("Synthetic configurations differ between the original records")
    return reference, recorded


def trials(cfg):
    return [
        (seed, method, lr)
        for seed in cfg["seeds"]
        for method in cfg["methods"]
        for lr in cfg["lrs"]
    ]


def key(seed, method, lr):
    return f"fresh_s{seed}_{method}_lr{int(lr * 10**7)}"


def prepare(root, code):
    root, code = root.resolve(), code.resolve()
    if root == code or root in code.parents or code in root.parents:
        raise ValueError("Use an output directory outside the source tree")
    reference, recorded = specification(code)
    identity = dict(
        schema="ggf-fresh-zoo-v1",
        configuration=reference["config"],
        source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_sources={
            p.relative_to(code).as_posix(): sha256(p.read_bytes()).hexdigest()
            for folder in [code / "training/core", code / "evidence/zoo_guided/source"]
            for p in sorted(folder.rglob("*.py"))
        },
        generator_sha256=sha256(
            (code / "evidence/zoo_guided/source/zoo_benchmark/data.py").read_bytes()
        ).hexdigest(),
        baseline_versions=reference["original_versions"],
        guided_versions=recorded["versions"],
    )
    marker = root / "FRESH_ZOO.json"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise ValueError("Choose an empty directory or an existing fresh Zoo run")
    if marker.exists() and read(marker) != identity:
        raise ValueError("Fresh Zoo configuration/source identity changed")
    root.mkdir(parents=True, exist_ok=True)
    if not marker.exists():
        save(marker, identity)
    return reference, identity


def check_versions(expected):
    from importlib.metadata import version

    actual = {name: version(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            "Install the recorded phase environment: " + json.dumps(expected)
        )


def environment(expected, device):
    import torch

    check_versions(expected)
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Training and evaluation require a CUDA runtime")
    from threadpoolctl import threadpool_limits

    torch.set_num_threads(2)
    threadpool_limits(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def generate(root, code):
    import torch

    reference, identity = prepare(root, code)
    check_versions(identity["baseline_versions"])
    focused, _, data = modules(code)
    path = root / "dataset.pt"
    if path.exists():
        load_data(root)
        return
    if (root / "DATA.json").exists():
        raise FileNotFoundError("Dataset payload is missing")
    dataset = data.generate(
        reference["config"], "goodfire", "independent", reference["config"]["data_seed"]
    )
    audit = data.audit(dataset)
    temporary = root / "dataset.pt.tmp"
    torch.save(dataset, temporary)
    temporary.replace(path)
    save(
        root / "DATA.json",
        dict(
            sha256=sha256(path.read_bytes()).hexdigest(),
            fingerprints=focused.fingerprints(dataset),
            generator_checks=audit,
            matches_original_arrays=focused.fingerprints(dataset)
            == reference["original_data_fingerprints"],
        ),
    )


def load_data(root):
    import torch

    path = root / "dataset.pt"
    if sha256(path.read_bytes()).hexdigest() != read(root / "DATA.json")["sha256"]:
        raise ValueError("Generated dataset checksum changed")
    return torch.load(path, map_location="cpu", weights_only=False)


def store(root, native, cfg, identity, name):
    return native.Store(
        root / name,
        cfg,
        sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
        local=root / "scratch" / name,
        require_drive=False,
    )


def train_baselines(root, code, device):
    reference, identity = prepare(root, code)
    environment(identity["baseline_versions"], device)
    _, native, _ = modules(code)
    dataset = load_data(root)
    cfg = reference["config"]
    journal = store(root, native, cfg, identity, "baselines")
    selected = []
    for seed in cfg["seeds"]:
        for method in cfg["methods"]:
            states = [
                native.train_native(
                    journal,
                    key(seed, method, lr),
                    method,
                    dataset["packs"]["train"]["x"],
                    dataset["packs"]["select"]["x"],
                    seed,
                    lr,
                    device,
                )
                for lr in cfg["lrs"]
            ]
            expected = (cfg["train_rows"] // cfg["batch"]) * cfg["epochs"]
            if any(state["updates"] != expected for state in states):
                raise ValueError("Incomplete baseline trial")
            choice = native.choose_native(journal, seed, method, states)
            selected.append(
                {
                    k: choice[k]
                    for k in [
                        "method",
                        "seed",
                        "weights_sha",
                        "lr",
                        "chosen_field",
                        "sparsity_matched",
                    ]
                }
            )
            del states, choice
            gc.collect()
    save(
        root / "BASELINES_COMPLETE.json",
        dict(
            complete=True,
            selection=selected,
            data_sha256=read(root / "DATA.json")["sha256"],
        ),
    )


def train_guided(root, code, device):
    reference, identity = prepare(root, code)
    environment(identity["guided_versions"], device)
    focused, native, _ = modules(code)
    cfg = reference["config"]
    journal = store(root, native, cfg, identity, "guided")
    dataset = load_data(root)
    catalog = []
    for seed in cfg["seeds"]:
        candidates = []
        for lr in cfg["lrs"]:
            name = key(seed, "ggf", lr)
            state = focused.fit(journal, name, cfg, dataset, seed, lr, device=device)
            if state["updates"] != (cfg["train_rows"] // cfg["batch"]) * cfg["epochs"]:
                raise ValueError("Incomplete guided trial")
            candidates.append(
                dict(
                    key=name,
                    lr=lr,
                    selection_mse=state["best_mse"],
                    best_update=state["best_update"],
                    updates=state["updates"],
                    weights_sha=focused.weight_sha(state["best"]),
                    final_sha=focused.weight_sha(state["weights"]),
                )
            )
            del state
            gc.collect()
        catalog.append(
            dict(
                seed=seed,
                **min(candidates, key=lambda t: (t["selection_mse"], t["lr"])),
                trials=candidates,
            )
        )
    journal.save("zoo_selected", catalog)
    save(
        root / "GUIDED_COMPLETE.json",
        dict(
            complete=True,
            selection=catalog,
            data_sha256=read(root / "DATA.json")["sha256"],
        ),
    )


def require_fits(root, cfg):
    digest = read(root / "DATA.json")["sha256"]
    expected_baselines = {(s, m) for s in cfg["seeds"] for m in cfg["methods"]}
    for file, expected in [
        ("BASELINES_COMPLETE.json", expected_baselines),
        ("GUIDED_COMPLETE.json", set(cfg["seeds"])),
    ]:
        value = read(root / file)
        actual = (
            {(r["seed"], r["method"]) for r in value["selection"]}
            if file.startswith("BASELINES")
            else {r["seed"] for r in value["selection"]}
        )
        if (
            not value["complete"]
            or value["data_sha256"] != digest
            or actual != expected
            or len(value["selection"]) != len(expected)
        ):
            raise ValueError("Incomplete or mismatched synthetic fits")


def audit(root, code, guided, device):
    reference, identity = prepare(root, code)
    cfg = reference["config"]
    require_fits(root, cfg)
    environment(identity["guided_versions" if guided else "baseline_versions"], device)
    focused, native, _ = modules(code)
    dataset = load_data(root)
    journal = store(root, native, cfg, identity, "guided" if guided else "baselines")
    rows = []
    if guided:
        for choice in read(root / "GUIDED_COMPLETE.json")["selection"]:
            state = journal.load(choice["key"])
            for field, label, expected in [
                ("best", "ggf_selected", choice["weights_sha"]),
                ("weights", "ggf_final_diagnostic", choice["final_sha"]),
            ]:
                if focused.weight_sha(state[field]) != expected:
                    raise ValueError("Guided weights differ from the selection receipt")
                result = focused.audit(
                    journal,
                    f"audit_s{choice['seed']}_{label}",
                    cfg,
                    dataset,
                    state[field],
                    device,
                )
                rows.append(dict(seed=choice["seed"], method=label, **result))
    else:
        for choice in read(root / "BASELINES_COMPLETE.json")["selection"]:
            selected = journal.load(f"s{choice['seed']}_{choice['method']}_selected")
            if native.sha_weights(selected["weights"]) != choice["weights_sha"]:
                raise ValueError("Baseline weights differ from the selection receipt")
            result = native.audit(
                journal,
                f"s{choice['seed']}_{choice['method']}",
                selected["weights"],
                choice["method"],
                dataset,
                device,
            )
            rows.append(dict(seed=choice["seed"], method=choice["method"], **result))
    save(
        root / ("GUIDED_RESULTS.json" if guided else "BASELINE_RESULTS.json"),
        dict(
            complete=True,
            models=native.serial(rows),
            data_sha256=read(root / "DATA.json")["sha256"],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--code-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "generate",
            "train-baselines",
            "train-ggf",
            "audit-baselines",
            "audit-ggf",
        ],
        required=True,
    )
    args = parser.parse_args()
    root, code = args.root.resolve(), args.code_root.resolve()
    actions = {
        "generate": lambda: generate(root, code),
        "train-baselines": lambda: train_baselines(root, code, "cuda"),
        "train-ggf": lambda: train_guided(root, code, "cuda"),
        "audit-baselines": lambda: audit(root, code, False, "cuda"),
        "audit-ggf": lambda: audit(root, code, True, "cuda"),
    }
    actions[args.phase]()


if __name__ == "__main__":
    main()
