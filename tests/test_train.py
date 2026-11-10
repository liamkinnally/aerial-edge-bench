"""Failure records must not leave a stopped run marked as running."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class TrainingFailureTest(unittest.TestCase):
    def test_checkpoint_load_failure_is_recorded(self):
        root = Path(__file__).resolve().parents[1]
        original = subprocess.check_output

        def metadata(command, **kwargs):
            if command[:2] == ["sysctl", "-n"]:
                return "1024\n" if command[-1] == "hw.memsize" else "test CPU\n"
            return original(command, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            argv = ["train.py", "--epochs", "1", "--output", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "path", [str(root / "scripts"), *sys.path]),
                patch("torch.backends.mps.is_available", return_value=True),
                patch("subprocess.check_output", side_effect=metadata),
                patch("ultralytics.YOLO", side_effect=RuntimeError("checkpoint failed")),
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint failed"):
                    runpy.run_path(str(root / "scripts/train.py"), run_name="__main__")
            result = json.loads((output / "run-metadata.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertIn("checkpoint failed", result["error"])
            self.assertIn("finished_at_utc", result)


if __name__ == "__main__":
    unittest.main()
