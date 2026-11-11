"""Failure records must not leave a stopped run marked as running."""

import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class TrainingFailureTest(unittest.TestCase):
    def test_memory_batch_reduction_stops_before_recorder_restart(self):
        root = Path(__file__).resolve().parents[1]
        original = subprocess.check_output

        def metadata(command, **kwargs):
            if command[:2] == ["sysctl", "-n"]:
                return "1024\n" if command[-1] == "hw.memsize" else "test CPU\n"
            return original(command, **kwargs)

        recorders = []

        class Recorder:
            def __init__(self, *args, **kwargs):
                self.epochs = []
                self.starts = 0
                self.closed = False
                recorders.append(self)

            def on_epoch_start(self, trainer):
                if self.starts:
                    raise RuntimeError("Recorder received a duplicate epoch start")
                self.starts += 1

            def on_fit_epoch_end(self, trainer):
                pass

            def close(self):
                self.closed = True

        class Model:
            def __init__(self):
                self.callbacks = {}

            def add_callback(self, event, callback):
                self.callbacks.setdefault(event, []).append(callback)

            def train(self, **configuration):
                self.configuration = configuration
                trainer = SimpleNamespace(
                    epoch=0, batch_size=16, _oom_retries=0,
                    train_loader=SimpleNamespace(batch_size=16),
                )
                for callback in self.callbacks["on_train_epoch_start"]:
                    callback(trainer)
                trainer.batch_size = trainer.train_loader.batch_size = 8
                trainer._oom_retries = 1
                for callback in self.callbacks["on_train_epoch_start"]:
                    callback(trainer)

        model = Model()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            argv = ["train.py", "--epochs", "50", "--batch", "16", "--output", str(output)]
            with (
                patch.object(sys, "argv", argv),
                patch.object(sys, "path", [str(root / "scripts"), *sys.path]),
                patch("torch.backends.mps.is_available", return_value=True),
                patch("subprocess.check_output", side_effect=metadata),
                patch("ultralytics.YOLO", return_value=model),
                patch("training_records.EpochRecorder", Recorder),
            ):
                with self.assertRaisesRegex(RuntimeError, "Batch changed.*memory recovery"):
                    runpy.run_path(str(root / "scripts/train.py"), run_name="__main__")
            result = json.loads((output / "run-metadata.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["batch_change"], {
                "requested_batch": 16, "observed_batch": 8,
                "trainer_batch": 8, "epoch": 1, "oom_retries": 1,
            })
            self.assertEqual(model.configuration["epochs"], 50)
            self.assertEqual(model.configuration["batch"], 16)
            self.assertFalse(model.configuration["resume"])
            self.assertEqual(recorders[0].starts, 1)
            self.assertTrue(recorders[0].closed)
            self.assertEqual(recorders[0].epochs, [])
            self.assertIn("finished_at_utc", result)

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
