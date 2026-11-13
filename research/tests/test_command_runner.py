"""Child-process diagnostics remain visible and are retained after failure."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "command_runner", ROOT / "code/bootstrap/command_runner.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CommandRunnerTests(unittest.TestCase):
    def test_failure_preserves_first_exception_and_both_output_channels(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            redirect_stdout(io.StringIO()) as out,
        ):
            command = [
                sys.executable,
                "-c",
                "print('stage started'); raise ValueError('underlying cause')",
            ]
            with self.assertRaisesRegex(RuntimeError, "ValueError: underlying cause"):
                MODULE.run_command(command, temp, "smoke")
            log = next(Path(temp).glob("smoke_*.log")).read_text(encoding="utf-8")
            for expected in (
                "stage started",
                "Traceback",
                "ValueError: underlying cause",
            ):
                self.assertIn(expected, log)
                self.assertIn(expected, out.getvalue())

    def test_retries_keep_separate_logs(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            command = [sys.executable, "-c", "print('complete')"]
            for _ in range(2):
                self.assertEqual(
                    MODULE.run_command(command, temp, "check").returncode, 0
                )
            logs = list(Path(temp).glob("check_*.log"))
            self.assertEqual(len(logs), 2)
            self.assertTrue(
                all(p.read_text(encoding="utf-8") == "complete\n" for p in logs)
            )


if __name__ == "__main__":
    unittest.main()
