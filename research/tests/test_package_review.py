"""Regression checks for directly browsable, complete release archives."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "package_review", ROOT / "scripts/package_review.py"
)
PACKAGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACKAGE)


class PackageReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        self.output = Path(self.temporary.name) / "release"
        for folder in (
            "code/frozen_sources",
            "code/tools",
            "code/training",
            "code/vendor",
            "code/runs",
            "code/figures",
            "notebooks",
            "docs",
            "scripts",
            "tests",
            "paper/cvpr2027",
        ):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            ROOT / "code/tools/materialize_frozen.py",
            self.root / "code/tools/materialize_frozen.py",
        )
        self.source = b"def example():\n    return 75\n"
        self.identity = sha256(self.source).hexdigest()
        vault = self.root / "code/frozen_sources/blobs.zip"
        with ZipFile(vault, "w", ZIP_DEFLATED) as archive:
            archive.writestr(self.identity, self.source)
        profile = {
            "blob_archives": {
                "frozen_sources/blobs.zip": sha256(vault.read_bytes()).hexdigest()
            },
            "profiles": {
                "matched75_original": {
                    "entrypoint": "method.py",
                    "pythonpath": ["."],
                    "files": {
                        "method.py": {
                            "blob_archive": "frozen_sources/blobs.zip",
                            "sha256": self.identity,
                            "bytes": len(self.source),
                        }
                    },
                }
            },
        }
        (self.root / "code/frozen_sources/profiles.json").write_text(
            json.dumps(profile)
        )
        (self.root / "code/APPENDIX_PATHS.json").write_text('["training/model.py"]')
        for relative in ("code/training/model.py", "code/vendor/model.py"):
            (self.root / relative).write_bytes(self.source)
        for relative in (
            "README.md",
            "LICENSE",
            "LICENSES.md",
            "requirements.txt",
            "docs/REPRODUCING.md",
            "docs/METHODS.md",
            "scripts/verify_bundle.py",
            "scripts/package_review.py",
            "tests/test_methods.py",
            "scripts/verify_source_presentation.py",
            "docs/VERIFICATION.json",
        ):
            (self.root / relative).write_text("# fixture\n")
        (self.root / "paper/cvpr2027/ggf_cvpr2027_supplement.pdf").write_bytes(
            b"%PDF-test"
        )
        (self.root / "code/runs/scratch.py").write_text("raise RuntimeError()")
        (self.root / "code/figures/generated.pdf").write_bytes(b"generated")
        cache = self.root / "code/training/__pycache__"
        cache.mkdir()
        (cache / "model.pyc").write_bytes(b"cache")

    def test_build_physically_stores_duplicate_and_historical_sources(self):
        standalone = PACKAGE.build(self.root, self.output)
        for path in (
            standalone,
            self.output / "cvpr2027/GGF_CVPR2027_supplementary.zip",
        ):
            with self.subTest(archive=path.name), ZipFile(path) as archive:
                names = set(archive.namelist())
                self.assertNotIn("FILE_ALIASES.json", names)
                for name in (
                    "code/training/model.py",
                    "code/vendor/model.py",
                    "code/sources/matched75_original/method.py",
                ):
                    self.assertEqual(archive.read(name), self.source)
                self.assertIn("docs/METHODS.md", names)
                self.assertIn("tests/test_methods.py", names)
                self.assertIn("scripts/verify_source_presentation.py", names)
                self.assertIn("docs/VERIFICATION.json", names)
                self.assertNotIn("code/runs/scratch.py", names)
                self.assertNotIn("code/figures/generated.pdf", names)
                self.assertFalse(any("__pycache__" in name for name in names))
                manifest = json.loads(archive.read("FILES.json"))
                self.assertEqual(names, set(manifest) | {"FILES.json"})
                for name, expected in manifest.items():
                    self.assertEqual(sha256(archive.read(name)).hexdigest(), expected)
                self.assertIsNone(archive.testzip())

    def test_build_refuses_to_replace_changed_historical_source(self):
        PACKAGE.materialize_profiles(self.root)
        path = self.root / "code/sources/matched75_original/method.py"
        path.write_text("# retained local edits\n")
        with self.assertRaises(FileExistsError):
            PACKAGE.build(self.root, self.output)
        self.assertEqual(path.read_text(), "# retained local edits\n")
        self.assertFalse(self.output.exists())

    def test_missing_cited_evidence_stops_build(self):
        (self.root / "code/training/model.py").unlink()
        with self.assertRaises(FileNotFoundError):
            PACKAGE.build(self.root, self.output)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
