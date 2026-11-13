"""Guard the cleanup audit against accidentally ignoring executable changes."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "presentation_audit", ROOT / "scripts/verify_source_presentation.py"
)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SourcePresentationTests(unittest.TestCase):
    def test_whitespace_and_comments_are_allowed(self):
        before = '"""AGF training."""\ndef fit(x):\n return x+1 # AGF\n'
        after = '"""AGF training."""\n\ndef fit(x):\n    return x + 1  # GGF\n'
        self.assertEqual(AUDIT.normalized_tree(before), AUDIT.normalized_tree(after))

    def test_numerical_objective_change_is_detected(self):
        self.assertNotEqual(
            AUDIT.normalized_tree("def loss(x): return x * 0.4"),
            AUDIT.normalized_tree("def loss(x): return x * 0.5"),
        )

    def test_runtime_storage_names_are_not_normalized(self):
        self.assertNotEqual(
            AUDIT.normalized_tree('STORE = "AGF_COCO"'),
            AUDIT.normalized_tree('STORE = "GGF_COCO"'),
        )

    def test_call_order_and_identifiers_are_not_normalized(self):
        self.assertNotEqual(
            AUDIT.normalized_tree("a(); b()"), AUDIT.normalized_tree("b(); a()")
        )
        self.assertNotEqual(
            AUDIT.normalized_tree("AGF = 1"), AUDIT.normalized_tree("GGF = 1")
        )


if __name__ == "__main__":
    unittest.main()
