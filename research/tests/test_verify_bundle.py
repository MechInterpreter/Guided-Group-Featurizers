"""Check source availability and integrity before any reproduction work."""

from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_bundle", ROOT / "scripts/verify_bundle.py"
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


class VerifyBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = b"def method():\n    return 75\n"
        self.path = self.root / "code/sources/matched75_original/method.py"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(source)
        manifest = self.root / "code/frozen_sources/profiles.json"
        manifest.parent.mkdir()
        manifest.write_text(
            json.dumps(
                {
                    "profiles": {
                        "matched75_original": {
                            "files": {
                                "method.py": {
                                    "bytes": len(source),
                                    "sha256": sha256(source).hexdigest(),
                                },
                            }
                        }
                    }
                }
            )
        )

    def test_directly_browsable_source_is_verified_without_writes(self):
        before = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(VERIFY.verify_browsable_profiles(self.root), 1)
        after = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_missing_source_fails_instead_of_being_restored(self):
        self.path.unlink()
        with self.assertRaises(FileNotFoundError):
            VERIFY.verify_browsable_profiles(self.root)
        self.assertFalse(self.path.exists())

    def test_changed_source_fails(self):
        self.path.write_text("# altered source\n")
        with self.assertRaises(ValueError):
            VERIFY.verify_browsable_profiles(self.root)

    def test_unsafe_paths_are_rejected(self):
        for name in (
            "../outside.py",
            "/outside.py",
            "C:/outside.py",
            "code\\outside.py",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                VERIFY.bundle_path(self.root, name)


if __name__ == "__main__":
    unittest.main()
