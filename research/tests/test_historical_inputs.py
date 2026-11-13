"""Public-input lineage, directory ownership and audit gates for older studies."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE / "bootstrap"))
import historical_inputs as inputs


class HistoricalInputTests(unittest.TestCase):
    def test_reserved_audits_keep_the_original_distinct_allocations(self):
        controls = inputs.audit_ids(CODE)
        five = inputs.audit_ids(CODE, "supervised-five")
        self.assertEqual(len(controls), 1500)
        self.assertEqual(len(five), 1500)
        self.assertNotEqual(controls, five)
        spec = inputs.read(CODE / "bootstrap/SPECIFICATION.json")
        self.assertEqual(controls, spec["splits"]["audit"])
        for split in ["train", "select", "confirm"]:
            self.assertFalse(set(five).intersection(spec["splits"][split]))

    def test_output_owner_preserves_nonowned_inputs_and_refuses_source_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = root / "code"
            code.mkdir()
            data = root / "data"
            data.mkdir()
            (data / "keep.txt").write_text("keep")
            driver = code / "fit.py"
            driver.write_text("pass")
            run = root / "run"
            inputs.own(run, code, [data], driver, "fixture")
            inputs.own(run, code, [data], driver, "fixture")
            driver.write_text("changed")
            with self.assertRaisesRegex(ValueError, "source or dependency"):
                inputs.own(run, code, [data], driver, "fixture")
            with self.assertRaisesRegex(ValueError, "empty output"):
                inputs.own(data, code, [], driver, "fixture")
            with self.assertRaisesRegex(ValueError, "separate"):
                inputs.own(data / "inside", code, [data], driver, "fixture")

    def test_audit_rejects_incomplete_catalog_before_creating_artifacts(self):
        inputs.activate(CODE)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            for catalog in [
                {},
                dict(complete=False, models=[1]),
                dict(complete=True, models=[]),
            ]:
                with self.assertRaisesRegex(ValueError, "complete model catalog"):
                    inputs.audit_pack({}, CODE, root, "controls", catalog)
            self.assertFalse(root.exists())

    def test_only_declared_dependency_keys_are_read_through(self):
        own = SimpleNamespace(load=lambda key: None)
        upstream = SimpleNamespace(load=lambda key: "upstream " + key)
        view = inputs.InputView(own, {"allowed": upstream})
        self.assertEqual(view.load("allowed"), "upstream allowed")
        self.assertIsNone(view.load("other"))

    def test_input_path_cannot_escape_its_store(self):
        with self.assertRaisesRegex(ValueError, "leaves"):
            inputs.artifact(ROOT, "../outside")

    def test_control_catalog_freezes_primary_before_extra_fits(self):
        import fresh_controls

        events = []
        primary = ["ggf", "vanilla", "supervised"]
        extra = ["extra"]
        experiment = SimpleNamespace(
            SEEDS=[21, 22, 23],
            PRIMARY=primary,
            EXTRA=extra,
            endpoint=lambda ctx, seed, kind: (
                events.append((seed, kind)) or dict(seed=seed, kind=kind)
            ),
            validate_row=lambda row: None,
            freeze=lambda ctx, rows, name: events.append((name, len(rows))),
        )
        with patch.dict(sys.modules, {"experiment": experiment}):
            fresh_controls.train({})
        self.assertEqual(events[9], ("primary_frozen", 9))
        self.assertEqual(events[-1], ("all_models_frozen", 12))
        with patch.dict(sys.modules, {"experiment": experiment}):
            with self.assertRaisesRegex(ValueError, "Incomplete or duplicate"):
                fresh_controls.validate_catalog(
                    [dict(seed=21, kind="ggf")] * 3, primary
                )

    def test_feature_bytes_and_training_lineage_are_checked(self):
        inputs.activate(CODE)
        import numpy as np
        import torch
        from coco_baselines.storage import identity

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = root / "code"
            (code / "bootstrap").mkdir(parents=True)
            (code / "training").mkdir()
            (code / "training/model.py").write_text("pass")
            (code / "bootstrap/fresh_matched75.py").write_text("pass")
            spec = dict(splits=dict(train=[1, 2], select=[3], confirm=[4]))
            inputs.write(code / "bootstrap/SPECIFICATION.json", spec)
            source = root / "source"
            folder = source / "features"
            folder.mkdir(parents=True)
            marker = dict(
                specification_sha256=inputs.sha(code / "bootstrap/SPECIFICATION.json"),
                implementation_sha256=inputs.sha(code / "bootstrap/fresh_matched75.py"),
                training_source_sha256=identity(
                    {"training/model.py": inputs.sha(code / "training/model.py")}
                ),
            )
            inputs.write(source / "FRESH_RUN.json", marker)
            torch.save(
                dict(mean=np.zeros((196, 768), np.float32), scale=1.0),
                folder / "normalization.pt",
            )
            index = dict(
                run=marker,
                normalization=dict(
                    path="normalization.pt",
                    sha256=inputs.sha(folder / "normalization.pt"),
                ),
                packs={},
            )
            for split, ids in spec["splits"].items():
                np.save(
                    folder / (split + ".npy"),
                    np.zeros((len(ids), 196, 768), np.float32),
                )
                torch.save(
                    dict(ids=ids, masks=[{} for _ in ids]), folder / (split + ".pt")
                )
                index["packs"][split] = dict(
                    array=split + ".npy",
                    array_sha256=inputs.sha(folder / (split + ".npy")),
                    masks=split + ".pt",
                    masks_sha256=inputs.sha(folder / (split + ".pt")),
                    ids=ids,
                    identity="fixture-" + split,
                )
            inputs.write(folder / "FEATURES.json", index)
            _, _, _, packs = inputs.development_inputs(code, source)
            self.assertEqual(packs["select"]["ids"], [3])
            for pack in packs.values():
                pack["x"]._mmap.close()
            with (folder / "select.npy").open("ab") as file:
                file.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "Feature checksum"):
                inputs.development_inputs(code, source)
            (code / "training/model.py").write_text("different")
            with self.assertRaisesRegex(ValueError, "training-source identity"):
                inputs.development_inputs(code, source)


if __name__ == "__main__":
    unittest.main()
