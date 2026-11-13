"""Notebook setup installs matching binary wheels before testing image imports."""

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]


class NotebookEnvironmentTests(unittest.TestCase):
    def source(self):
        notebook = json.loads((ROOT / "notebooks/reproduce_paper.ipynb").read_bytes())
        return next(
            "".join(c["source"])
            for c in notebook["cells"]
            if c["cell_type"] == "code"
            and "install_torch_family" in "".join(c["source"])
        )

    def test_matched_cuda_wheels_are_always_checked_including_when_torch_is_installed(
        self,
    ):
        calls = []
        scope = dict(
            json=json,
            CODE=ROOT / "code",
            REPO=ROOT,
            LOGS=ROOT / "unused_logs",
            sys=sys,
            RUN_MODE="quick",
            run_command=lambda *a: calls.append(a),
        )
        exec(self.source(), scope)
        self.assertEqual(len(calls), 3)
        install = calls[0][0]
        for wheel in (
            "torch==2.11.0+cu130",
            "torchvision==0.26.0+cu130",
            "torchaudio==2.11.0+cu130",
        ):
            self.assertIn(wheel, install)
        self.assertEqual(
            install[-2:], ["--index-url", "https://download.pytorch.org/whl/cu130"]
        )
        self.assertEqual(calls[-1][-1], "check_image_imports")

    def test_probe_executes_compiled_nms_and_imports_transformers_in_a_new_process(
        self,
    ):
        calls = []
        exec(
            self.source(),
            dict(
                json=json,
                CODE=ROOT / "code",
                REPO=ROOT,
                LOGS=ROOT / "unused_logs",
                sys=sys,
                RUN_MODE="quick",
                run_command=lambda *a: calls.append(a),
            ),
        )
        probe = calls[-1][0]
        self.assertEqual(probe[:2], [sys.executable, "-c"])
        compile(probe[2], "image_import_check", "exec")
        self.assertIn("keep = nms(", probe[2])
        self.assertIn(
            "from transformers import AutoImageProcessor, AutoModel", probe[2]
        )

    def test_evidence_mode_does_not_install_gpu_dependencies(self):
        calls = []
        exec(
            self.source(),
            dict(RUN_MODE="evidence", run_command=lambda *a: calls.append(a)),
        )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
