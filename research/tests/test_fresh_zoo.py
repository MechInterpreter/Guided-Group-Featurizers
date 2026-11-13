"""Fresh synthetic-run configuration, allocation and checkpoint contracts."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fresh_zoo", ROOT / "code/bootstrap/fresh_zoo.py"
)
ZOO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ZOO)


class FreshZooTests(unittest.TestCase):
    def test_exact_paper_configuration_and_trial_matrix(self):
        reference, _ = ZOO.specification(ROOT / "code")
        cfg = reference["config"]
        self.assertEqual(
            (cfg["dimension"], cfg["factors"], cfg["groups"], cfg["block_rank"]),
            (128, 128, 256, 4),
        )
        self.assertEqual(len(ZOO.trials(cfg)), 18)
        self.assertEqual(cfg["epochs"], 200)

    def test_source_identity_and_nonempty_directory_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            ZOO.prepare(root, ROOT / "code")
            first = (root / "FRESH_ZOO.json").read_bytes()
            ZOO.prepare(root, ROOT / "code")
            self.assertEqual(first, (root / "FRESH_ZOO.json").read_bytes())
            value = ZOO.read(root / "FRESH_ZOO.json")
            value["configuration"]["groups"] = 999
            ZOO.save(root / "FRESH_ZOO.json", value)
            with self.assertRaises(ValueError):
                ZOO.prepare(root, ROOT / "code")
            with self.assertRaises(ValueError):
                ZOO.prepare(Path(folder), ROOT / "code")

    def test_audit_requires_every_fit_and_matching_generated_data(self):
        cfg = {"seeds": [0, 1], "methods": ["vanilla", "grassmannian"]}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ZOO.save(root / "DATA.json", {"sha256": "data"})
            ZOO.save(
                root / "BASELINES_COMPLETE.json",
                dict(
                    complete=True,
                    data_sha256="data",
                    selection=[
                        dict(seed=s, method=m)
                        for s in cfg["seeds"]
                        for m in cfg["methods"]
                    ],
                ),
            )
            ZOO.save(
                root / "GUIDED_COMPLETE.json",
                dict(
                    complete=True,
                    data_sha256="data",
                    selection=[dict(seed=s) for s in cfg["seeds"]],
                ),
            )
            ZOO.require_fits(root, cfg)
            bad = ZOO.read(root / "GUIDED_COMPLETE.json")
            bad["selection"].pop()
            ZOO.save(root / "GUIDED_COMPLETE.json", bad)
            with self.assertRaises(ValueError):
                ZOO.require_fits(root, cfg)

    def test_original_native_trainer_runs_and_resumes_without_drive(self):
        try:
            import numpy as np
            import torch
            import scipy
            import einops
        except ImportError as error:
            self.skipTest(str(error))
        _, native, _ = ZOO.modules(ROOT / "code")
        cfg = dict(
            dimension=8,
            groups=8,
            block_rank=2,
            active_factors=2,
            batch=16,
            epochs=2,
            warmup=0.05,
            snr=0.0,
            log_epochs=1,
            save_updates=2,
        )
        x = np.random.default_rng(1).normal(size=(32, 8)).astype("float32")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            outputs = []
            for name, resume in [("whole", False), ("resumed", True)]:
                journal = ZOO.store(root, native, cfg, {"fixture": "CPU"}, name)
                if resume:
                    native.train_native(
                        journal, "fit", "vanilla", x, x, 0, 0.0004, "cpu", max_updates=2
                    )
                result = native.train_native(
                    journal, "fit", "vanilla", x, x, 0, 0.0004, "cpu"
                )
                self.assertEqual(result["updates"], 4)
                outputs.append(native.sha_weights(result["weights"]))
            self.assertEqual(outputs[0], outputs[1])

    def test_original_guided_trainer_runs_without_drive(self):
        try:
            import torch
            import numpy
            import scipy
            import einops
        except ImportError as error:
            self.skipTest(str(error))
        focused, native, data = ZOO.modules(ROOT / "code")
        cfg = dict(
            dimension=16,
            factors=8,
            groups=16,
            block_rank=4,
            active_factors=4,
            train_rows=64,
            select_rows=32,
            calibration_rows=32,
            test_rows=32,
            normalization_rows=64,
            batch=16,
            epochs=2,
            warm_epochs=1,
            warmup=0.05,
            log_epochs=1,
            snr=0.0,
            save_updates=2,
        )
        dataset = data.generate(cfg, "goodfire", "independent", seed=701)
        with tempfile.TemporaryDirectory() as folder:
            journal = ZOO.store(
                Path(folder), native, cfg, {"fixture": "guided CPU"}, "guided"
            )
            result = focused.fit(journal, "fit", cfg, dataset, 0, 0.0004, device="cpu")
            self.assertEqual(result["updates"], 8)
            self.assertTrue(torch.isfinite(result["head"]).all())
            self.assertTrue(result["history"][-1]["phase"] == "Guided Learning")


if __name__ == "__main__":
    unittest.main()
