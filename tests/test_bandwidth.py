"""Guard clip-rate selection, payload aggregation and saved-result preservation."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
os.environ["YOLO_AUTOINSTALL"] = "false"

import cv2
import numpy as np
from scripts import bandwidth


class BandwidthTests(unittest.TestCase):
    def test_clip_fps_and_all_frame_payloads_are_saved(self):
        # Catches hard-coded FPS, dropping empty detections, last-frame-only
        # aggregation, and missing/mislabelled persistence at the CLI boundary.
        self.assertTrue(hasattr(bandwidth, "main"), "Bandwidth CLI is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            clip = directory / "fixture.mp4"
            output = directory / "result.json"
            writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
            self.assertTrue(writer.isOpened())
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
            writer.write(np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8))
            writer.release()
            capture = cv2.VideoCapture(str(clip))
            jpeg_sizes = []
            for _ in range(2):
                ok, frame = capture.read()
                self.assertTrue(ok)
                ok, jpg = cv2.imencode(".jpg", frame)
                self.assertTrue(ok)
                jpeg_sizes.append(len(jpg))
            capture.release()
            box = SimpleNamespace(cls=1, conf=0.876, xyxy=np.array([[1.2, 2.6, 10.9, 12.1]]))
            detector = unittest.mock.Mock(side_effect=[
                [SimpleNamespace(boxes=[])], [SimpleNamespace(boxes=[box])],
            ])
            detector.predictor = SimpleNamespace(args=SimpleNamespace(imgsz=640, conf=0.25))
            weights = directory / "fixture.pt"
            weights.write_bytes(b"stubbed detector weights")
            argv = ["bandwidth.py", "--model", str(weights), "--clip", str(clip),
                    "--fps", "30", "--output", str(output)]
            stdout = io.StringIO()
            with patch.object(sys, "argv", argv), patch("ultralytics.YOLO", return_value=detector), \
                    contextlib.redirect_stdout(stdout):
                bandwidth.main()
            record = json.loads(output.read_text())
            # Exact messages are [] and [{"cls": 1, "conf": 0.88, "box": [1, 3, 11, 12]}].
            self.assertEqual(record["sizes_b"], [2, 49])
            self.assertEqual(record["sizes_a"], jpeg_sizes)
            self.assertEqual(record["frame_count"], 2)
            self.assertEqual(record["FPS"], 24)
            self.assertEqual(record["requested_fps"], 30)
            self.assertEqual(record["resolution"], {"width": 64, "height": 64})
            self.assertEqual(record["jpeg_quality"], 95)
            self.assertEqual(record["avg_b"], 25.5)
            self.assertEqual(record["per_sec_b"], 612)
            self.assertEqual(record["avg_a"], sum(jpeg_sizes) / 2)
            self.assertEqual(record["per_sec_a"], sum(jpeg_sizes) * 12)
            self.assertAlmostEqual(record["ratio"], sum(jpeg_sizes) / 51)
            self.assertEqual(record["baseline"], "JPEG-frame payload bandwidth")
            self.assertIn("24", stdout.getvalue())
            self.assertIn(record["summary_line"], stdout.getvalue())
            # File bytes over this fixture's 1/12-second duration, not FPS=30,
            # and file payload divided by the detection payload for the ratio.
            self.assertEqual(record["clip_file_size_bytes"], clip.stat().st_size)
            self.assertEqual(record["clip_duration_seconds"], 1 / 12)
            self.assertAlmostEqual(record["per_sec_clip"], clip.stat().st_size * 12)
            self.assertAlmostEqual(record["ratio_clip"], clip.stat().st_size / 51)
            self.assertEqual(record["clip_baseline"], "average MP4 file bytes per second")
            self.assertIn(record["clip_summary_line"], stdout.getvalue())

    def test_existing_result_is_not_overwritten_or_remeasured(self):
        # Prevents an accidental repeated command from destroying its evidence.
        self.assertTrue(hasattr(bandwidth, "main"), "Bandwidth CLI is not implemented")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text('{"keep": true}\n')
            with patch.object(sys, "argv", ["bandwidth.py", "--output", str(output)]), \
                    patch("ultralytics.YOLO", side_effect=AssertionError("must not run")):
                with self.assertRaises(FileExistsError):
                    bandwidth.main()
            self.assertEqual(output.read_text(), '{"keep": true}\n')


if __name__ == "__main__":
    unittest.main()
