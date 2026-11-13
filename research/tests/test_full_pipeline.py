"""Prerequisite ordering, resumability and completion gates for fresh studies."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/bootstrap"))
import full_pipeline as pipeline
import fresh_legacy as legacy


class FullPipelineTests(unittest.TestCase):
    def test_feature_cache_is_only_used_for_development_extraction(self):
        previous = ROOT.parent / "previous_run/matched75"
        tasks = pipeline.plan(ROOT / "code", ROOT.parent / "fixture_run", previous)
        extraction = tasks["matched.extract"]["command"]
        self.assertIn("--feature-cache", extraction)
        self.assertIn(str(previous), extraction)
        self.assertTrue(
            all(
                "--feature-cache" not in task["command"]
                for name, task in tasks.items()
                if name != "matched.extract"
            )
        )

    def test_plan_has_every_study_and_closes_all_prerequisites(self):
        tasks = pipeline.plan(ROOT / "code", ROOT.parent / "fixture_run")
        ordered = pipeline.ordered(tasks, tasks)
        self.assertEqual(len(ordered), len(tasks))
        positions = {name: i for i, name in enumerate(ordered)}
        for name, task in tasks.items():
            self.assertTrue(Path(task["command"][1]).is_file())
            for dependency in task["dependencies"]:
                self.assertLess(positions[dependency], positions[name])
        for stage in legacy.DEPENDENCIES:
            self.assertIn("legacy." + stage + ".evaluate", tasks)
        self.assertIn("analysis.controlled", tasks)
        self.assertIn("supervised5.evaluate", tasks)
        self.assertIn("concept.voc.evaluate", tasks)
        self.assertEqual(tasks["matched.coherence"]["annotated"], True)
        self.assertNotIn("lighting", " ".join(ordered).lower())
        self.assertNotIn("blender", " ".join(ordered).lower())

    def test_requested_voc_compute_includes_coco_selection_and_both_voc_fits(self):
        tasks = pipeline.plan(ROOT / "code", ROOT.parent / "fixture_run")
        names = pipeline.ordered(tasks, ["compute"])
        for required in [
            "concept.coco.train",
            "voc19.train",
            "controls.voc.train",
            "concept.voc.train",
        ]:
            self.assertIn(required, names)
        self.assertNotIn("matched.coherence", names)

    def test_cycles_and_missing_stages_fail(self):
        with self.assertRaisesRegex(ValueError, "Unknown"):
            pipeline.ordered({}, ["missing"])
        with self.assertRaisesRegex(ValueError, "Cyclic"):
            pipeline.ordered({"a": {"dependencies": ["a"]}}, ["a"])

    def test_synthetic_generation_uses_the_recorded_environment_and_audits_wait_for_both_fits(
        self,
    ):
        tasks = pipeline.plan(ROOT / "code", ROOT.parent / "fixture_run")
        self.assertIn("zoo_baselines", tasks["zoo.generate"]["command"][0])
        self.assertEqual(
            tasks["zoo.generate"]["command"][0],
            tasks["zoo.train-baselines"]["command"][0],
        )
        for stage in ["zoo.audit-baselines", "zoo.audit-ggf"]:
            names = pipeline.ordered(tasks, [stage])
            self.assertIn("zoo.train-baselines", names)
            self.assertIn("zoo.train-ggf", names)
            self.assertLess(names.index("zoo.train-baselines"), names.index(stage))
            self.assertLess(names.index("zoo.train-ggf"), names.index(stage))

    def test_nonpaper_renderer_is_absent_from_the_deliverable_sources(self):
        manifest = json.loads((ROOT / "code/frozen_sources/profiles.json").read_bytes())
        files = manifest["profiles"]["controlled_variation"]["files"]
        for name in ["lighting_study.py", "lighting_math.py", "render_lighting.py"]:
            self.assertNotIn(name, files)
            self.assertFalse(
                (ROOT / "code/sources/controlled_variation" / name).exists()
            )
        registry = json.loads((ROOT / "code/workflow_registry.json").read_bytes())
        self.assertNotIn("frozen.controlled_variation.run", registry)
        notebook = json.loads((ROOT / "notebooks/reproduce_paper.ipynb").read_bytes())
        source = "".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn("full_pipeline.py", source)
        self.assertNotIn("lighting", source.lower())
        self.assertNotIn("blender", source.lower())

    def test_failed_child_never_gets_a_completion_and_success_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            code, root = base / "code", base / "run"
            (code / "bootstrap").mkdir(parents=True)
            (code / "worker.py").write_text("pass\n")
            tasks = {
                name: dict(
                    command=["worker", name],
                    dependencies=[],
                    annotated=False,
                    receipt=str(root / (name + ".json")),
                )
                for name in ["one", "two"]
            }
            calls = []

            def fail(command, *_):
                calls.append(command[-1])
                if command[-1] == "two":
                    raise RuntimeError("worker failed")
                pipeline.write(root / "one.json", dict(complete=True, count=3))

            with patch.object(pipeline, "run_command", fail):
                with self.assertRaisesRegex(RuntimeError, "worker failed"):
                    pipeline.run(root, code, tasks, ["one", "two"])
            self.assertTrue((root / "progress/one.json").exists())
            self.assertFalse((root / "progress/two.json").exists())
            self.assertFalse((root / "PIPELINE_RUN_COMPLETE.json").exists())
            calls.clear()

            def finish(command, *_):
                calls.append(command[-1])
                pipeline.write(root / "two.json", dict(complete=True))

            with patch.object(pipeline, "run_command", finish):
                pipeline.run(root, code, tasks, ["one", "two"])
            self.assertEqual(calls, ["two"])
            (root / "one.json").write_text('{"complete":true,"count":4}')
            with self.assertRaisesRegex(ValueError, "output changed"):
                pipeline.run(root, code, tasks, ["one"])

    def test_audit_extension_preserves_development_receipt_but_locks_audit(self):
        features = dict(
            run={"pin": "fixed"},
            normalization={"sha": "fixed"},
            packs={key: {"sha": key} for key in ["train", "select", "confirm"]},
        )
        before = pipeline.receipt_digest("matched.extract", features)
        features.update(audit_training_receipts={"full": "fit"})
        features["packs"]["audit"] = {"sha": "audit-a"}
        self.assertEqual(before, pipeline.receipt_digest("matched.extract", features))
        audit = pipeline.receipt_digest("matched.audit", features)
        features["packs"]["audit"]["sha"] = "audit-b"
        self.assertNotEqual(audit, pipeline.receipt_digest("matched.audit", features))

    def test_historical_kernels_and_width_growth_execute_on_small_cpu_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "quick"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code/bootstrap/quick_pipeline.py"),
                    "--root",
                    str(root),
                    "--device",
                    "cpu",
                    "--phase",
                    "historical",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((root / "checks/historical.json").read_bytes())
            self.assertTrue(report["complete"])
            self.assertNotIn("lighting", report)


if __name__ == "__main__":
    unittest.main()
