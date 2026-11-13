"""VOC input allocation, normalization and fresh-run guards."""

import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
SPEC = importlib.util.spec_from_file_location(
    "fresh_voc", ROOT / "code/bootstrap/fresh_voc.py"
)
VOC = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VOC)


class FreshVOCTests(unittest.TestCase):
    def test_recorded_voc_batch_and_evaluation_configuration(self):
        specification = VOC.read(ROOT / "code/bootstrap/SPECIFICATION.json")
        before = specification["fit_config"].copy()
        cfg = VOC.fit_configuration(specification)
        self.assertEqual(cfg["batch"], 512)
        self.assertEqual(cfg["audit_image_batch"], 1)
        self.assertEqual(cfg["bootstrap"], 1000)
        self.assertEqual(cfg["preview_images"], 0)
        self.assertEqual(specification["fit_config"], before)

    def test_new_directory_and_source_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "fresh"
            _, first = VOC.prepare_root(root, ROOT / "code")
            self.assertEqual(VOC.prepare_root(root, ROOT / "code")[1], first)
            with self.assertRaises(ValueError):
                VOC.prepare_root(Path(temp), ROOT / "code")
            changed = dict(first, implementation_sha256="changed")
            VOC.write(root / "FRESH_VOC.json", changed)
            with self.assertRaises(ValueError):
                VOC.prepare_root(root, ROOT / "code")

    def test_exact_model_and_revision(self):
        calls = []
        loader = VOC.pinned_loader(
            lambda *a, **kw: calls.append((a, kw)),
            {"model": "pinned-model", "revision": "pinned-revision"},
        )
        loader("pinned-model", revision="latest", dtype="float32")
        self.assertEqual(calls[0][1]["revision"], "pinned-revision")
        self.assertEqual(calls[0][1]["dtype"], "float32")
        with self.assertRaises(ValueError):
            loader("other-model")

    def test_six_completed_fits_required(self):
        rows = [
            dict(seed=s, arm=m, selected_total_updates=10640, groups=19)
            for s in [21, 22, 23]
            for m in ["GGF", "SLiCS"]
        ]
        VOC.validate_rows(rows)
        for altered in [
            rows[:-1],
            rows + [rows[0]],
            [dict(r, groups=75) for r in rows],
            [dict(r, selected_total_updates=7640) for r in rows],
        ]:
            with self.assertRaises(ValueError):
                VOC.validate_rows(altered)

    def source_functions(self):
        try:
            import numpy as np
        except ImportError as error:
            self.skipTest(str(error))
        path = ROOT / "code/sources/voc_slics_compute/voc_training.py"
        tree = ast.parse(path.read_bytes())
        definitions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name in {"partition", "streaming_norm"}
        ]
        scope = {"np": np, "hashlib": __import__("hashlib")}
        exec(
            compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"),
            scope,
        )
        return scope, np

    def test_original_partition_keeps_official_validation_separate(self):
        source, _ = self.source_functions()
        rows = [
            dict(image=f"train_{i}", official_split="train") for i in range(510)
        ] + [dict(image=f"eval_{i}", official_split="val") for i in range(10)]
        partition = source["partition"](rows)
        self.assertEqual(list(partition.values()).count("select"), 256)
        self.assertEqual(list(partition.values()).count("train"), 254)
        self.assertTrue(all(partition[f"eval_{i}"] == "eval" for i in range(10)))
        self.assertEqual(partition, source["partition"](list(reversed(rows))))

    def test_original_normalization_uses_fit_images_only(self):
        source, np = self.source_functions()
        raw = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[3, 4, 5], [6, 7, 8]],
                [[20, 30, 40], [50, 60, 70]],
            ],
            dtype="float16",
        )
        norm = source["streaming_norm"](raw, [0, 1])
        np.testing.assert_array_equal(norm["mean"], [[2, 3, 4], [5, 6, 7]])
        self.assertAlmostEqual(norm["scale"], 1.0)
        raw[2] *= 10
        repeated = source["streaming_norm"](raw, [0, 1])
        self.assertEqual(norm["train_sha"], repeated["train_sha"])
        np.testing.assert_array_equal(norm["mean"], repeated["mean"])
        self.assertEqual(norm["scale"], repeated["scale"])


if __name__ == "__main__":
    unittest.main()
