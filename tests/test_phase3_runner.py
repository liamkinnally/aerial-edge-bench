import csv
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


class MatrixRunnerTest(unittest.TestCase):
    def test_native_validation_and_unchanged_loop_feed_one_row(self):
        scripts = Path(__file__).resolve().parents[1] / "scripts"
        with tempfile.TemporaryDirectory() as directory, patch.object(sys, "path", [str(scripts), *sys.path]):
            from phase3_runtime import ShapeAudit
            import torch
            from ultralytics import settings

            previous_dataset_directory = settings["datasets_dir"]

            root = Path(directory)
            weights = root / "results/phase1-fifty-epochs/weights/best.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"checkpoint fixture")
            image = root / "image.jpg"
            image.write_bytes(b"image fixture")
            output, csv_file = root / "run", root / "matrix.csv"
            validation_audit, timing_audit = ShapeAudit(640), ShapeAudit(640)
            for audit, count in ((validation_audit, 548), (timing_audit, 110)):
                audit.shapes.add((1, 3, 640, 640))
                audit.count = count
            aggregate = {"metrics/mAP50-95(B)": 0.2, "metrics/mAP50(B)": 0.4}
            classes = [{"Class": "pedestrian", "mAP50-95": 0.1, "mAP50": 0.3}]
            metrics = SimpleNamespace(results_dict=aggregate, to_json=lambda **_: json.dumps(classes))
            validator = SimpleNamespace(shape_audit=validation_audit, seen=548, torch_threads=8,
                                        args=SimpleNamespace(rect=False, imgsz=640))
            predictor = SimpleNamespace(shape_audit=timing_audit, args=SimpleNamespace(rect=False, imgsz=640),
                                        model=SimpleNamespace(device=torch.device("cpu"), fp16=False))
            timed = Mock(return_value=(20.0, 20.0, 50.0, [20.0] * 100,
                                       SimpleNamespace(predictor=predictor), SimpleNamespace(shape=(1080, 1920, 3))))
            argv = ["benchmark_phase3.py", "--model", str(weights), "--format", "pytorch_fp32", "--size", "640",
                    "--image", str(image), "--output", str(output), "--csv", str(csv_file)]
            with (
                patch.object(sys, "argv", argv),
                patch("phase3_common.ROOT", root),
                patch("phase3_common.configure_environment"),
                patch("ultralytics.settings.update"),
                patch("phase3_common.provenance", return_value={"python": "test", "library_versions": {"torch": "test"},
                                                               "git_commit": "test-commit", "hardware": "test CPU"}),
                patch("phase3_runtime.make_guarded_types", return_value=(Mock(), SimpleNamespace(last_instance=validator))),
                patch("ultralytics.YOLO", return_value=SimpleNamespace(val=lambda **_: metrics)),
                patch("timing.measure_latency", timed),
                patch("psutil.cpu_percent", return_value=0.0),
                patch("torch.set_num_interop_threads"),
            ):
                runpy.run_path(str(scripts / "benchmark_phase3.py"), run_name="__main__")
            self.assertEqual(settings["datasets_dir"], previous_dataset_directory)
            with csv_file.open() as handle:
                row, = list(csv.DictReader(handle))
            self.assertEqual(row["val_map50_95"], str(aggregate["metrics/mAP50-95(B)"]))
            self.assertEqual(row["pedestrian_ap50_95"], str(classes[0]["mAP50-95"]))
            record = json.loads((output / "run.json").read_text())
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["validation_shapes"]["calls"], 548)
            self.assertEqual(record["timing_shapes"]["calls"], 110)
            timed.assert_called_once_with(str(weights), str(image), 100)


if __name__ == "__main__":
    unittest.main()
