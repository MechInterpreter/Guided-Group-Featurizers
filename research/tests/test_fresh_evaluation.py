"""CPU dispatch contracts for fresh concentration; no model imports or services."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "code/bootstrap/fresh_evaluation.py"
spec = importlib.util.spec_from_file_location("fresh_evaluation_contract", MODULE)
fresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fresh)


class FreshPresenceStages(unittest.TestCase):
    def setUp(self):
        self.root = Path("fresh-run")
        self.calls = []
        self.read = Mock(
            return_value={
                "complete": True,
                "scope": "fresh_public_data_matched75",
            }
        )
        self.eval_module = types.SimpleNamespace(read=self.read)
        self.preparation = types.SimpleNamespace(
            lock_presence=lambda root, role: self.calls.append(("lock", root, role)),
            presence_quality=lambda root: self.calls.append(("qa", root)),
        )
        self.scoring = types.SimpleNamespace(
            select_presence=lambda root: self.calls.append(("select", root)),
            concentration=lambda root: self.calls.append(("concentration", root)),
        )
        self.modules = patch.dict(
            sys.modules,
            {
                "eval_common": self.eval_module,
                "prepare_evaluation_reviews": self.preparation,
                "score_evaluation": self.scoring,
            },
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_lock_both_presence_splits_then_independent_qa(self):
        fresh.presence_stage(self.root, "lock-presence")
        out = self.root / "evaluation"
        self.assertEqual(
            self.calls,
            [
                ("lock", out, "calibration"),
                ("lock", out, "audit"),
                ("qa", out),
            ],
        )
        self.read.assert_called_once_with(out / "COLLECTION_COMPLETE.json")

    def test_qa_failure_propagates(self):
        self.preparation.presence_quality = Mock(
            side_effect=ValueError("QA incomplete")
        )
        with self.assertRaisesRegex(ValueError, "QA incomplete"):
            fresh.presence_stage(self.root, "lock-presence")
        self.assertEqual([call[0] for call in self.calls], ["lock", "lock"])

    def test_selector_is_separate_from_audit_scoring(self):
        fresh.presence_stage(self.root, "select-presence")
        self.assertEqual(self.calls, [("select", self.root / "evaluation")])

    def test_concentration_does_not_reselect(self):
        fresh.presence_stage(self.root, "concentration")
        self.assertEqual(self.calls, [("concentration", self.root / "evaluation")])

    def test_incomplete_and_historical_receipts_rejected_before_helpers(self):
        for receipt in [
            {"complete": False, "scope": "fresh_public_data_matched75"},
            {"complete": True, "scope": "historical"},
        ]:
            with self.subTest(receipt=receipt):
                self.read.return_value = receipt
                with self.assertRaisesRegex(ValueError, "fresh native collection"):
                    fresh.presence_stage(self.root, "concentration")
        self.assertEqual(self.calls, [])

    def test_missing_labels_are_not_imputed(self):
        self.scoring.select_presence = Mock(
            side_effect=FileNotFoundError("presence labels")
        )
        with self.assertRaises(FileNotFoundError):
            fresh.presence_stage(self.root, "select-presence")
        self.assertEqual(self.calls, [])

    def test_cli_dispatches_only_requested_stage(self):
        argv = [
            str(MODULE),
            "--root",
            str(self.root),
            "--code-root",
            str(ROOT / "code"),
            "--phase",
            "concentration",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(sys, "path", sys.path.copy()),
            patch.object(fresh, "presence_stage") as stage,
        ):
            fresh.main()
        stage.assert_called_once_with(self.root.resolve(), "concentration")
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
