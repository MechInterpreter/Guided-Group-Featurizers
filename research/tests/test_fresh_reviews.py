"""Fresh whole-group scoring preserves evidence, denominators and source locks."""

from contextlib import ExitStack
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/bootstrap"))
import fresh_reviews as reviews

reviews.setup(ROOT / "code")
import prepare_evaluation_reviews as preparation
import score_evaluation


class FreshReviewTests(unittest.TestCase):
    def test_majority_preserves_unknowns_and_three_way_disagreement(self):
        self.assertEqual(
            reviews.majority(["clear", "partial", "no_clear_theme"]), "unresolved_vote"
        )
        self.assertEqual(
            reviews.majority(["insufficient_evidence"] * 2 + ["clear"]),
            "insufficient_evidence",
        )

    def test_frozen_content_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "raw.json"
            reviews.freeze(p, {"value": "first"})
            reviews.freeze(p, {"value": "first"})
            with self.assertRaisesRegex(ValueError, "changed"):
                reviews.freeze(p, {"value": "second"})
            self.assertEqual(reviews.read(p), {"value": "first"})

    def fixture(self, root):
        out = root / "evaluation"
        pair, whole = out / "reviews/coherence", out / "reviews/whole"
        reviews.freeze(pair / "REVIEW_LOCK.json", {})
        public = dict(
            cases=[
                dict(
                    case_id=str(i),
                    valid_high_views=list(range(1, 7)),
                    valid_high_count=6,
                )
                for i in range(75)
            ]
        )
        private = dict(
            cases=[dict(case_id=str(i), bank_id="bank", unit=i) for i in range(75)],
            banks=[
                dict(
                    bank_id="bank",
                    seed=21,
                    regime="full_category",
                    kind="fixture",
                    fold="all",
                )
            ],
        )
        reviews.freeze(whole / "PUBLIC.json", public)
        reviews.freeze(whole / "PRIVATE.json", private)
        reviews.freeze(
            whole / "REVIEW_LOCK.json",
            dict(
                public_sha256=reviews.digest(whole / "PUBLIC.json"),
                private_sha256=reviews.digest(whole / "PRIVATE.json"),
                pair_lock_sha256=reviews.digest(pair / "REVIEW_LOCK.json"),
                rubric_source_sha256=reviews.instructions(ROOT / "code")[1],
            ),
        )
        for j, context in enumerate(reviews.CONTEXTS):
            rows = []
            for i in range(75):
                insufficient = i == 74
                rows.append(
                    dict(
                        case_id=str(i),
                        theme="striped fabric" if not insufficient else "",
                        fit_views=[] if insufficient else list(range(1, 7)),
                        high_fit_count=0 if insufficient else 6,
                        exceptions=list(range(1, 7)) if insufficient else [],
                        coherence="insufficient_evidence" if insufficient else "clear",
                        specificity="unclear" if insufficient else "specific",
                        confidence="low" if insufficient else "high",
                        high_low_contrast="not_assessed",
                        observations="Fixture evidence.",
                    )
                )
            reviews.freeze(
                whole / "labels" / (context + ".json"),
                dict(
                    public_sha256=reviews.digest(whole / "PUBLIC.json"),
                    method_blind=True,
                    reviewer="fixture context " + str(j),
                    cases=rows,
                ),
            )
        return out, whole

    def test_unknowns_count_in_denominator_and_rubric_source_is_bound(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            out, folder = self.fixture(root)
            stack.enter_context(
                patch.object(preparation, "verify_pack", return_value={})
            )
            pair_score = stack.enter_context(
                patch.object(score_evaluation, "coherence")
            )
            reviews.score(root, ROOT / "code")
            result = reviews.read(out / "WHOLE_COHERENCE_COMPLETE.json")
            self.assertEqual(result["banks"][0]["groups"], 75)
            self.assertEqual(result["banks"][0]["clear_rate"], 74 / 75)
            self.assertEqual(result["banks"][0]["counts"]["insufficient_evidence"], 1)
            self.assertEqual(len(result["group_outcomes"]), 75)
            pair_score.assert_called_once()
            instructions = reviews.instructions(ROOT / "code")
            with patch.object(
                reviews, "instructions", return_value=(instructions[0], "changed")
            ):
                with self.assertRaisesRegex(ValueError, "lock changed"):
                    reviews.score(root, ROOT / "code")

    def test_missing_case_fails_before_scoring_or_complete_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, folder = self.fixture(root)
            path = folder / "labels/context1.json"
            raw = copy.deepcopy(reviews.read(path))
            raw["cases"].pop()
            path.write_text(__import__("json").dumps(raw))
            with patch.object(preparation, "verify_pack", return_value={}):
                with self.assertRaisesRegex(ValueError, "Every planned"):
                    reviews.score(root, ROOT / "code")
            self.assertFalse((out / "WHOLE_COHERENCE_COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
