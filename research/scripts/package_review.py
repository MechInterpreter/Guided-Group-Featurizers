"""Build self-contained reviewer ZIPs with directly browsable source files."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import runpy
from zipfile import ZIP_DEFLATED, ZipFile


def digest(data):
    """Return the SHA-256 digest used by the delivered file manifests."""
    return sha256(data).hexdigest()


def materialize_profiles(root):
    """Expose each exact historical study at ``code/sources/<profile>``.

    The source vaults remain unchanged. Their existing materializer verifies every
    blob and refuses to replace a modified file before writing any profile files.
    Only manifest-listed files are included in the release.
    """
    code = root / "code"
    profiles = json.loads((code / "frozen_sources/profiles.json").read_bytes())
    materialize = runpy.run_path(str(code / "tools/materialize_frozen.py"))[
        "materialize"
    ]
    selected = []
    for name, profile in sorted(profiles["profiles"].items()):
        destination = code / "sources" / name
        if destination.resolve().parent != (code / "sources").resolve():
            raise ValueError(f"Unsafe source profile name: {name!r}")
        materialize(name, destination)
        selected.extend(destination / relative for relative in profile["files"])
    return selected


def payload(root, profile_files=()):
    """Select maintained code, verified historical sources and submission docs."""
    selected = []
    generated_roots = tuple(
        root / name
        for name in (
            "code/figures",
            "code/sec",
            "code/runs",
            "code/sources",
        )
    )
    ignored_parts = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    for folder in ("code", "notebooks", "tests"):
        selected.extend(
            path
            for path in (root / folder).rglob("*")
            if path.is_file()
            and not ignored_parts.intersection(path.relative_to(root).parts)
            and path.suffix not in {".pyc", ".pyo"}
            and not any(path.is_relative_to(excluded) for excluded in generated_roots)
        )
    selected.extend(profile_files)
    selected.extend((root / "docs").glob("*.md"))
    selected.extend(
        root / name
        for name in (
            "README.md",
            "LICENSE",
            "LICENSES.md",
            "requirements.txt",
            "scripts/verify_bundle.py",
            "scripts/package_review.py",
        )
    )
    for name in (
        "pyproject.toml",
        "requirements-dev.txt",
        "scripts/verify_source_presentation.py",
        "docs/VERIFICATION.json",
    ):
        if (root / name).is_file():
            selected.append(root / name)
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in selected}


def validate_appendix_paths(files):
    """Require every cited evidence path to be physically stored in the ZIP."""
    for path in json.loads(files["code/APPENDIX_PATHS.json"]):
        full = "code/" + path
        present = (
            any(name.startswith(full) for name in files)
            if path.endswith("/")
            else full in files
        )
        if not present:
            raise FileNotFoundError(f"Missing cited Appendix path: {full}")


def make_archive(path, files):
    """Write and read back an archive before atomically replacing its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=8) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("Built archive failed its CRC check")
        for name, raw in files.items():
            if digest(archive.read(name)) != digest(raw):
                raise ValueError(f"Built archive changed {name}")
    os.replace(temporary, path)


def build(root, output):
    """Build the standalone repository and supplementary ZIP without aliases."""
    root, output = Path(root).resolve(), Path(output).resolve()
    supplement = root / "paper/cvpr2027/ggf_cvpr2027_supplement.pdf"
    supplement_bytes = supplement.read_bytes()
    files = payload(root, materialize_profiles(root))
    validate_appendix_paths(files)
    code = {
        name.removeprefix("code/"): raw
        for name, raw in files.items()
        if name.startswith("code/") and name != "code/SHA256SUMS"
    }
    checksums = "".join(
        f"{digest(raw)}  {name}\n" for name, raw in sorted(code.items())
    ).encode()
    (root / "code/SHA256SUMS").write_bytes(checksums)
    files["code/SHA256SUMS"] = checksums
    manifest = {name: digest(raw) for name, raw in sorted(files.items())}
    files["FILES.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    standalone = output / "ggf_anonymous_repository.zip"
    make_archive(standalone, files)

    files[supplement.name] = supplement_bytes
    manifest[supplement.name] = digest(files[supplement.name])
    files["FILES.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    make_archive(output / "cvpr2027/GGF_CVPR2027_supplementary.zip", files)
    return standalone


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = build(args.root, args.output or args.root / "release")
    print(
        f"Built {target.name} and the CVPR supplementary ZIP with directly browsable sources."
    )
