"""Standalone notebook setup and ZIP-integrity checks."""

from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]


def setup_source(path):
    notebook = json.loads(path.read_bytes())
    source = next(
        "".join(c["source"])
        for c in notebook["cells"]
        if c["cell_type"] == "code" and "def locate_repository" in "".join(c["source"])
    )
    return source[: source.index("REPO = locate_repository()")]


@contextmanager
def directory(path):
    before = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(before)


class NotebookSetupTests(unittest.TestCase):
    def setUp(self):
        self.scope = {}
        exec(
            setup_source(ROOT / "notebooks/reproduce_paper.ipynb"),
            self.scope,
        )

    def archive(self, root, members=None, tamper=False):
        members = members or {"code/workflows.py": b"# workflow\n"}
        hashes = {name: sha256(raw).hexdigest() for name, raw in members.items()}
        with ZipFile(root / "ggf_anonymous_repository.zip", "w") as z:
            z.writestr("FILES.json", json.dumps(hashes))
            for name, raw in members.items():
                z.writestr(name, raw + b"changed" if tamper else raw)

    def test_detect_repository_from_root_and_notebooks_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "code").mkdir()
            (root / "code/workflows.py").write_text("")
            (root / "notebooks").mkdir()
            for cwd in [root, root / "notebooks"]:
                with directory(cwd):
                    self.assertEqual(self.scope["locate_repository"](), root.resolve())

    def test_extract_uploaded_archive(self):
        with tempfile.TemporaryDirectory() as temp, directory(temp):
            self.archive(Path(temp))
            root = self.scope["locate_repository"]()
            self.assertEqual((root / "code/workflows.py").read_bytes(), b"# workflow\n")

    def test_hash_failure_precedes_extraction(self):
        with tempfile.TemporaryDirectory() as temp, directory(temp):
            self.archive(Path(temp), tamper=True)
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.scope["locate_repository"]()
            self.assertFalse((Path(temp) / "ggf_repository").exists())

    def test_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as temp, directory(temp):
            self.archive(Path(temp), {"../escape.py": b"x"})
            with self.assertRaisesRegex(ValueError, "path"):
                self.scope["locate_repository"]()
            self.assertFalse((Path(temp) / "escape.py").exists())

    def test_existing_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp, directory(temp):
            self.archive(Path(temp), {"code/workflows.py": b"x", "README.md": b"old"})
            root = Path(temp) / "ggf_repository"
            root.mkdir()
            (root / "README.md").write_bytes(b"edited")
            with self.assertRaises(FileExistsError):
                self.scope["locate_repository"]()
            self.assertEqual((root / "README.md").read_bytes(), b"edited")

    def test_only_one_canonical_notebook(self):
        sources = [setup_source(p) for p in (ROOT / "notebooks").glob("*.ipynb")]
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(set(sources)), 1)


if __name__ == "__main__":
    unittest.main()
