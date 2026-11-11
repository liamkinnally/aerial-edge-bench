import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.phase3_records import (
    CSV_FIELDS, append_matrix_row, complete_matrix, extract_accuracy,
)


def result_row(model_format="pytorch_fp32", size=640):
    return {
        "format": model_format, "image_size": size,
        "actual_input_height": size, "actual_input_width": size,
        "val_map50_95": 0.2, "val_map50": 0.3,
        "pedestrian_ap50_95": 0.1, "pedestrian_ap50": 0.25,
        "mean_ms": 20.0, "p95_ms": 24.0, "fps": 50.0,
        "model_size_bytes": 1000000, "model_size_mb": 1.0,
        "device": "cpu", "thread_count": 8,
        "library_versions": "library=1", "git_commit": "test-commit",
        "hardware": 'CPU, "test machine"', "model_path": "model.pt",
        "image_path": "frame.jpg", "run_count": 100,
        "trained_image_size": 640,
        "size_experiment": "inference size only; trained at 640",
        "run_record": "run.json",
    }


class AccuracyTests(unittest.TestCase):
    def test_extracts_native_pedestrian_metrics_by_name(self):
        metrics = extract_accuracy(
            {"metrics/mAP50-95(B)": 0.2, "metrics/mAP50(B)": 0.3},
            [{"Class": "car", "mAP50-95": 0.8, "mAP50": 0.9},
             {"Class": "pedestrian", "mAP50-95": 0.1, "mAP50": 0.25}],
        )
        self.assertEqual(metrics, {
            "val_map50_95": 0.2, "val_map50": 0.3,
            "pedestrian_ap50_95": 0.1, "pedestrian_ap50": 0.25,
        })

    def test_incomplete_or_invalid_metrics_are_not_silently_filled(self):
        aggregate = {"metrics/mAP50-95(B)": 0.2, "metrics/mAP50(B)": 0.3}
        pedestrian = {"Class": "pedestrian", "mAP50-95": 0.1, "mAP50": 0.25}
        cases = [({}, [pedestrian]), (aggregate, []), (aggregate, [pedestrian, pedestrian]),
                 (aggregate, [{"Class": "pedestrian", "mAP50": 0.25}])]
        for bad in (float("nan"), float("inf"), -0.1, 1.1, True):
            cases.append(({**aggregate, "metrics/mAP50(B)": bad}, [pedestrian]))
            cases.append((aggregate, [{**pedestrian, "mAP50-95": bad}]))
        for inputs in cases:
            with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                extract_accuracy(*inputs)


class MatrixTests(unittest.TestCase):
    def test_six_distinct_rows_complete_and_quoted_values_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "matrix.csv"
            with self.assertRaises(ValueError):
                complete_matrix(path)
            expected = []
            for size in (640, 960):
                for model_format in ("pytorch_fp32", "onnx_fp32", "onnx_int8"):
                    row = result_row(model_format, size)
                    append_matrix_row(path, row)
                    expected.append({key: str(value) for key, value in row.items()})
                    if len(expected) < 6:
                        with self.assertRaises(ValueError):
                            complete_matrix(path)
            self.assertEqual(complete_matrix(path), expected)
            with path.open(newline="") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows[0], list(CSV_FIELDS))
            self.assertEqual(len(rows), 7)

    def test_shape_mismatch_or_corrupt_existing_shape_stops_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            append_matrix_row(path, result_row())
            original = path.read_bytes()
            for changes in ({"actual_input_height": 384}, {"actual_input_width": 960}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    append_matrix_row(path, {**result_row("onnx_fp32"), **changes})
                self.assertEqual(path.read_bytes(), original)
            corrupted = result_row("onnx_int8", 960)
            corrupted["actual_input_width"] = 640
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerow(corrupted)
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                append_matrix_row(path, result_row())
            with self.assertRaises(ValueError):
                complete_matrix(path)
            self.assertEqual(path.read_bytes(), original)

    def test_duplicate_or_invalid_rows_and_headers_preserve_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            append_matrix_row(path, result_row())
            original = path.read_bytes()
            valid = result_row("onnx_fp32")
            missing = valid.copy()
            del missing["git_commit"]
            invalid = [result_row(), missing, {**valid, "extra": "value"}]
            invalid.extend({**valid, **change} for change in (
                {"format": "unknown"}, {"image_size": 320}, {"device": "mps"},
                {"trained_image_size": 960}, {"size_experiment": "trained at 960"},
                {"thread_count": 0}, {"run_count": 1.5}, {"mean_ms": float("nan")},
                {"pedestrian_ap50_95": 2}, {"model_size_bytes": -1},
            ))
            for row in invalid:
                with self.subTest(row=row), self.assertRaises(ValueError):
                    append_matrix_row(path, row)
                self.assertEqual(path.read_bytes(), original)
            for content in ("wrong,header\nkeep,this\n", ",".join(CSV_FIELDS) + "\nshort,row\n"):
                path.write_text(content)
                original = path.read_bytes()
                with self.assertRaises(ValueError):
                    append_matrix_row(path, valid)
                self.assertEqual(path.read_bytes(), original)

    def test_failed_atomic_replacement_keeps_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            append_matrix_row(path, result_row())
            original = path.read_bytes()
            with patch.object(Path, "replace", side_effect=OSError("replacement failed")):
                with self.assertRaisesRegex(OSError, "replacement failed"):
                    append_matrix_row(path, result_row("onnx_fp32"))
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
