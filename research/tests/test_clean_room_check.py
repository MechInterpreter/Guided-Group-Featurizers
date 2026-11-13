"""Diagnostic safety contracts without network, GPU or model access."""

import importlib.util
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "clean_room_check", ROOT / "code/bootstrap/clean_room_check.py"
)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class CleanRoomTests(unittest.TestCase):
    def test_source_tree_and_ancestors_are_rejected(self):
        for path in [ROOT / "code", ROOT / "code/new", ROOT]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                CHECK.destination(path, ROOT / "code")

    def test_existing_inputs_are_not_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "checkpoint.pt").write_bytes(b"existing")
            with self.assertRaisesRegex(ValueError, "empty"):
                CHECK.destination(root, ROOT / "code")
            self.assertEqual((root / "checkpoint.pt").read_bytes(), b"existing")

    def test_only_training_images_selected(self):
        self.assertEqual(
            CHECK.training_images({"splits": {"train": [1, 2, 3], "audit": [4]}}),
            [1, 2],
        )

    def test_audit_and_incomplete_training_allocations_rejected(self):
        for train, audit in [([1, 2], [2]), ([1], [3]), ([1, 1], [3])]:
            with self.assertRaises(ValueError):
                CHECK.training_images({"splits": {"train": train, "audit": audit}})

    def test_preflight_checks_environment_before_model_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "new"
            module = types.SimpleNamespace(
                imports=lambda code: code,
                environment=Mock(side_effect=RuntimeError("CUDA required")),
                specification=Mock(),
                sha=Mock(),
            )
            with patch.dict("sys.modules", {"fresh_matched75": module}):
                with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                    CHECK.run(root, ROOT / "code", preflight_only=True)
            module.specification.assert_not_called()
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
