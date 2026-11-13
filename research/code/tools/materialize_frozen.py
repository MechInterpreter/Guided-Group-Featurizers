"""Restore a verified historical study without duplicating source snapshots."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

CODE = Path(__file__).resolve().parents[1]


def materialize(profile, output=None):
    manifest = json.loads((CODE / "frozen_sources/profiles.json").read_text())
    if profile not in manifest["profiles"]:
        raise ValueError("Unknown profile: " + profile)
    spec = manifest["profiles"][profile]
    contents = {}
    for archive_name in sorted({v["blob_archive"] for v in spec["files"].values()}):
        archive = CODE / archive_name
        if (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            != manifest["blob_archives"][archive_name]
        ):
            raise ValueError("Source vault hash differs: " + archive_name)
        with ZipFile(archive) as z:
            for name, record in spec["files"].items():
                if record["blob_archive"] != archive_name:
                    continue
                path = PurePosixPath(name)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or ":" in name
                    or "\\" in name
                ):
                    raise ValueError("Unsafe relative member: " + name)
                raw = z.read(record["sha256"])
                if (
                    len(raw) != record["bytes"]
                    or hashlib.sha256(raw).hexdigest() != record["sha256"]
                ):
                    raise ValueError("Source blob differs: " + name)
                contents[name] = raw
    if output is None:
        return {"profile": profile, "verified_files": len(contents)}
    destination = Path(output).resolve()
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(str(destination))
    # Check every existing target before any writes. Never replace changed work.
    for name, raw in contents.items():
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError("Output member escapes through a link: " + str(target))
        if target.exists() and not target.is_file():
            raise IsADirectoryError(str(target))
        if target.exists() and target.read_bytes() != raw:
            raise FileExistsError("Refusing to replace modified source: " + str(target))
    for name, raw in contents.items():
        target = destination / name
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as f:
            f.write(raw)
    receipt = {
        "profile": profile,
        "verified_files": len(contents),
        "entrypoint": spec["entrypoint"],
        "pythonpath": [str(destination / p) for p in spec["pythonpath"]],
        "scope": "Source restoration only. Supply licensed datasets, feature caches, checkpoints, and a study configuration before running.",
    }
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-all", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((CODE / "frozen_sources/profiles.json").read_text())
    if args.verify_all:
        result = [materialize(p) for p in manifest["profiles"]]
    elif args.profile:
        result = materialize(
            args.profile, args.output or CODE / "runs/sources" / args.profile
        )
    else:
        result = {p: v["description"] for p, v in manifest["profiles"].items()}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
