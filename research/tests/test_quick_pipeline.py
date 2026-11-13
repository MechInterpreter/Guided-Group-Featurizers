"""Completion gates for the minimum-work integration run."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/bootstrap"))
SPEC = importlib.util.spec_from_file_location(
    "quick_driver", ROOT / "code/bootstrap/quick_pipeline.py"
)
QUICK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUICK)


class QuickCompletionTests(unittest.TestCase):
    def code(self, path):
        code = path / "code"
        (code / "bootstrap").mkdir(parents=True)
        (code / "bootstrap/SPECIFICATION.json").write_text("{}")
        (code / "model.py").write_text("VALUE = 1\n")
        return code

    def arguments(self, code, run):
        return [
            "quick_pipeline.py",
            "--root",
            str(run),
            "--code-root",
            str(code),
            "--device",
            "cpu",
        ]

    def test_failed_section_cannot_create_complete_receipt_and_resume_skips_success(
        self,
    ):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = self.code(root)
            run = root / "run"
            visited = []

            def child(command, *_):
                stage = command[-1]
                visited.append(stage)
                if stage == "second":
                    raise RuntimeError("Child stage failed")
                QUICK.save(run / "checks" / (stage + ".json"), dict(complete=True))

            with (
                patch.object(QUICK, "STAGES", ("first", "second")),
                patch.object(sys, "argv", self.arguments(code, run)),
                patch.object(QUICK, "run_command", child),
            ):
                with self.assertRaisesRegex(RuntimeError, "Child stage failed"):
                    QUICK.main()
                self.assertFalse((run / "QUICK_COMPLETE.json").exists())
            self.assertEqual(visited, ["first", "second"])
            visited.clear()

            def finish(command, *_):
                visited.append(command[-1])
                QUICK.save(
                    run / "checks" / (command[-1] + ".json"), dict(complete=True)
                )

            with (
                patch.object(QUICK, "STAGES", ("first", "second")),
                patch.object(sys, "argv", self.arguments(code, run)),
                patch.object(QUICK, "run_command", finish),
            ):
                QUICK.main()
            receipt = json.loads((run / "QUICK_COMPLETE.json").read_bytes())
            self.assertEqual(visited, ["second"])
            self.assertTrue(receipt["complete"])
            self.assertFalse(receipt["research_run"])
            self.assertEqual(set(receipt["sections"]), {"first", "second"})

    def test_changed_source_cannot_reuse_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = self.code(root)
            QUICK.setup(root / "run", code, "cpu")
            (code / "model.py").write_text("VALUE = 2\n")
            with self.assertRaisesRegex(ValueError, "source or device changed"):
                QUICK.setup(root / "run", code, "cpu")

    def test_nonowned_and_source_directories_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = self.code(root)
            run = root / "run"
            run.mkdir()
            (run / "keep.txt").write_text("keep")
            with self.assertRaisesRegex(ValueError, "empty directory"):
                QUICK.setup(run, code, "cpu")
            with self.assertRaisesRegex(ValueError, "outside the source tree"):
                QUICK.setup(code / "run", code, "cpu")
            self.assertEqual((run / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
