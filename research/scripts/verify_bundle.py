"""Verify the directly browsable review payload and recount saved paper results.

Legacy alias restoration is retained only for compatibility with earlier bundles.
The current release stores every source file at its normal path.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import runpy
import subprocess
import sys
from zipfile import ZipFile


def bundle_path(root, name):
    path = root / name
    relative = PurePosixPath(name)
    if (
        not relative.parts
        or relative.is_absolute()
        or ".." in relative.parts
        or ":" in name
        or "\\" in name
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"Unsafe bundle path {name!r}")
    return path


def restore_aliases(root):
    index = root / "FILE_ALIASES.json"
    if not index.is_file():
        return
    hashes = json.loads((root / "FILES.json").read_bytes())
    if sha256(index.read_bytes()).hexdigest() != hashes.get(index.name):
        raise ValueError("Bundle alias manifest changed")
    aliases = json.loads(index.read_bytes())
    missing = []
    for name, ref in aliases.items():
        target = bundle_path(root, name)
        if target.exists():
            if (
                not target.is_file()
                or sha256(target.read_bytes()).hexdigest() != hashes[name]
            ):
                raise ValueError(f"Refusing to replace a modified file at {name}")
            continue
        source = ref["file"]
        if source in aliases or source not in hashes:
            raise ValueError(f"Alias source must be a stored bundle file for {name}")
        canonical = bundle_path(root, source)
        if "original" in ref:
            if (
                source != "code/frozen_legacy/exact_archives.py"
                or sha256(canonical.read_bytes()).hexdigest() != hashes[source]
            ):
                raise ValueError(
                    "Original archive restorer differs from the verified bundle"
                )
            data = runpy.run_path(str(canonical))["archive_bytes"](ref["original"])
        elif "member" in ref:
            with ZipFile(canonical) as vault:
                data = vault.read(ref["member"])
        else:
            data = canonical.read_bytes()
        if sha256(data).hexdigest() != hashes[name]:
            raise ValueError(f"Alias content hash differs for {name}")
        missing.append((target, data))
    for target, data in missing:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as file:
                file.write(data)
        except FileExistsError:
            if target.read_bytes() != data:
                raise
    print(f"Restored {len(missing)} shared file paths.")


def verify_browsable_profiles(root):
    """Check all delivered historical source files without restoring anything."""
    code = root / "code"
    manifest = json.loads((code / "frozen_sources/profiles.json").read_bytes())
    count = 0
    for profile, specification in manifest["profiles"].items():
        for name, record in specification["files"].items():
            relative = f"code/sources/{profile}/{name}"
            path = bundle_path(root, relative)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Historical source is not directly browsable: {relative}"
                )
            data = path.read_bytes()
            if (
                len(data) != record["bytes"]
                or sha256(data).hexdigest() != record["sha256"]
            ):
                raise ValueError(
                    f"Historical source differs from its frozen profile: {relative}"
                )
            count += 1
    print(
        f"Verified {count} directly browsable source files in {len(manifest['profiles'])} historical profiles."
    )
    return count


def verify(root):
    restore_aliases(root)
    manifest = root / "FILES.json"
    if manifest.is_file():
        hashes = json.loads(manifest.read_bytes())
        for name, expected in hashes.items():
            file = root / name
            assert file.resolve().is_relative_to(root.resolve())
            assert (
                file.is_file() and sha256(file.read_bytes()).hexdigest() == expected
            ), name
        print(f"Verified {len(hashes)} delivered file hashes.")
    hashes = root / "code/SHA256SUMS"
    for line in hashes.read_text().splitlines():
        expected, name = line.split("  ", 1)
        assert sha256((root / "code" / name).read_bytes()).hexdigest() == expected, name
    verify_browsable_profiles(root)
    presentation = root / "scripts/verify_source_presentation.py"
    if presentation.is_file():
        subprocess.run([sys.executable, str(presentation)], check=True)
    for script in ("verify_coherence.py", "verify_numeric_evidence.py"):
        subprocess.run([sys.executable, str(root / "code" / script)], check=True)
    subprocess.run(
        [sys.executable, str(root / "code/supporting_metrics/verify_support.py")],
        check=True,
    )
    legacy = root / "code/frozen_legacy"
    index = json.loads((legacy / "original_containers/index.json").read_bytes())
    archive_bytes = runpy.run_path(str(legacy / "exact_archives.py"))["archive_bytes"]
    for identity in index["archives"]:
        archive_bytes(identity)
    print(
        f"Verified {len(index['archives'])} exact source archives for the retained study profiles."
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "code/tools/materialize_frozen.py"),
            "--verify-all",
        ],
        check=True,
    )
    paths = json.loads((root / "code/APPENDIX_PATHS.json").read_bytes())
    for path in paths:
        file = root / "code" / path
        assert file.is_dir() if path.endswith("/") else file.is_file(), path
    print(f"Verified {len(paths)} Appendix evidence paths.")
    for file in (root / "notebooks").glob("*.ipynb"):
        notebook = json.loads(file.read_bytes())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["execution_count"] is None and not cell["outputs"], (
                    file.name
                )
                compile("".join(cell["source"]), file.name, "exec")
    print("Notebook code syntax and empty execution metadata verified.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--restore-only",
        action="store_true",
        help="Compatibility only: restore shared paths in older bundles.",
    )
    args = parser.parse_args()
    if args.restore_only:
        restore_aliases(args.root)
    else:
        verify(args.root)
