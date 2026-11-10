"""Integration guard for the installed predictor's setup and output behavior."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from scripts.predict import main


class PredictionIntegrationTest(unittest.TestCase):
    def test_requested_threads_survive_setup_and_video_is_saved(self):
        weights = ROOT / "models/yolo26n.pt"
        self.assertTrue(weights.is_file(), "Run the demo once to download its weights.")
        observed_threads = []

        def observed_model(path):
            model = YOLO(path)
            model.add_callback(
                "on_predict_batch_end",
                lambda predictor: observed_threads.append(torch.get_num_threads()),
            )
            return model

        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
            directory = Path(temporary)
            source = directory / "smoke.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.release()
            # Keep the project path relative to exercise Ultralytics path handling.
            project = (directory / "output").relative_to(Path.cwd())
            argv = [
                "predict.py", "--source", str(source), "--weights", str(weights),
                "--project", str(project), "--threads", "2", "--imgsz", "64",
            ]
            with patch.object(sys, "argv", argv), patch("ultralytics.YOLO", side_effect=observed_model):
                main()
            self.assertEqual(observed_threads, [2])
            outputs = list((directory / "output/phase0").glob("smoke.*"))
            self.assertEqual(len(outputs), 1)
            capture = cv2.VideoCapture(str(outputs[0]))
            try:
                self.assertTrue(capture.isOpened())
                self.assertTrue(capture.read()[0])
                self.assertFalse(capture.read()[0])
            finally:
                capture.release()


if __name__ == "__main__":
    unittest.main()
