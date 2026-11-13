"""Pinned upstream loading and all three released Concept-SAE objectives."""

from hashlib import sha256
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "code/training"),
    str(ROOT / "code/training/core"),
    str(ROOT / "code/sources/voc_slics_compute"),
]


class ConceptSourceTests(unittest.TestCase):
    def test_loader_pin_and_formatter_exclusion(self):
        import concept_sae as native

        path = ROOT / "code/training/concept_sae_native.py"
        self.assertEqual(sha256(path.read_bytes()).hexdigest(), native.SHA)
        self.assertIn(
            '"code/training/concept_sae_native.py"',
            (ROOT / "pyproject.toml").read_text(),
        )
        native.source(path)

    def test_three_original_objectives_train_and_produce_finite_outputs(self):
        import numpy as np
        import torch
        import concept_sae as native

        torch.manual_seed(21)
        model = native.build(
            native.source(ROOT / "code/training/concept_sae_native.py"),
            2,
            p=4,
            d=8,
            seg=2,
            free=2,
        )
        masks = []
        for _ in range(2):
            fraction = np.array(
                [[255, 0], [255, 0], [0, 255], [0, 255]], dtype=np.uint8
            )
            masks.append(dict(fraction=fraction, crowd=np.zeros(4)))
        pack = dict(
            x=np.random.default_rng(21).normal(size=(2, 4, 8)).astype("float32"),
            ids=[1, 2],
            masks=masks,
        )
        for stage in (1, 2, 3):
            part = native.configure(model, stage)
            optimizer = torch.optim.Adam(
                [p for p in part.parameters() if p.requires_grad], lr=0.001
            )
            value = native.optimizer_step(
                model, part, optimizer, pack, [0, 1], [0, 1], "cpu", stage, microbatch=1
            )
            self.assertTrue(np.isfinite(value))
        with torch.no_grad():
            prediction, _, _ = model(torch.from_numpy(pack["x"]))
        self.assertEqual(tuple(prediction.shape), (2, 4, 8))
        self.assertTrue(torch.isfinite(prediction).all())


if __name__ == "__main__":
    unittest.main()
