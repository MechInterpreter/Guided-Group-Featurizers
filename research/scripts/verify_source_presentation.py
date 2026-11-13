"""Verify that source presentation changes preserve the executable Python AST."""

import ast
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import inspect
from zipfile import ZipFile


def normalized_tree(raw):
    """Normalize only docstring indentation.

    Runtime string constants, identifiers, expressions and statement order are
    intentionally left untouched. Docstrings remain present in the syntax tree.
    """
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if node.body and isinstance(node.body[0], ast.Expr):
                value = node.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    value.value = inspect.cleandoc(value.value)
    return ast.dump(tree, include_attributes=False)


def verify(root):
    """Check archived originals and every maintained presentation file."""
    root = Path(root).resolve()
    provenance = root / "code/source_provenance"
    manifest = json.loads((provenance / "PRESENTATION.json").read_bytes())
    archive = provenance / "original_python.zip"
    if sha256(archive.read_bytes()).hexdigest() != manifest["original_archive_sha256"]:
        raise ValueError("Original-source archive differs")
    with ZipFile(archive) as original:
        if set(original.namelist()) != set(manifest["files"]):
            raise ValueError("Presentation archive membership differs")
        for name, record in manifest["files"].items():
            relative = PurePosixPath(name)
            path = root / name
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or ":" in name
                or "\\" in name
            ):
                raise ValueError("Unsafe source presentation path")
            if not path.resolve().is_relative_to(root):
                raise ValueError("Source presentation path escapes repository")
            before, after = original.read(name), path.read_bytes()
            if sha256(before).hexdigest() != record["original_sha256"]:
                raise ValueError("Original source differs: " + name)
            revision = manifest.get("functional_revisions", {}).get(name)
            if revision is not None:
                if (
                    revision["previous_presentation_sha256"]
                    != record["presentation_sha256"]
                    or sha256(after).hexdigest() != revision["current_sha256"]
                    or not (root / revision["regression_test"]).is_file()
                ):
                    raise ValueError("Declared functional revision differs: " + name)
                continue
            if sha256(after).hexdigest() != record["presentation_sha256"]:
                raise ValueError("Maintained source differs: " + name)
            if normalized_tree(before) != normalized_tree(after):
                raise ValueError("Executable AST differs: " + name)
    revised = len(manifest.get("functional_revisions", {}))
    count = len(manifest["files"]) - revised
    print(f"Verified executable AST equivalence for {count} presentation files.")
    if revised:
        print(
            f"Verified hashes for {revised} separately declared functional revisions."
        )
    return count


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
