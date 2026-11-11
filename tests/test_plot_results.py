"""Chart inputs must preserve measurement columns and comparable conditions."""

import csv
from pathlib import Path
import tempfile
import unittest

from scripts.phase3_records import CSV_FIELDS
from scripts.plot_results import load_chart_data


def matrix_rows():
    rows = []
    for index, (size, model_format) in enumerate(
        (size, model_format)
        for size in (640, 960)
        for model_format in ("pytorch_fp32", "onnx_fp32", "onnx_int8")
    ):
        row = dict.fromkeys(CSV_FIELDS, "recorded")
        row.update(
            format=model_format, image_size=size,
            actual_input_height=size, actual_input_width=size,
            val_map50_95=0.15 + index * 0.01, val_map50=0.3,
            pedestrian_ap50_95=0.1, pedestrian_ap50=0.2,
            mean_ms=10.5 + index, p95_ms=50.5 + index, fps=80.5 - index,
            model_size_bytes=3_000_000 + index * 100_000,
            model_size_mb=3.0 + index * 0.1,
            device="cpu", thread_count=8, run_count=100,
            trained_image_size=640,
            size_experiment="inference size only; trained at 640",
            hardware="Apple M4 Pro",
        )
        rows.append(row)
    return rows


def write_matrix(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class PlotDataTests(unittest.TestCase):
    def test_maps_mean_latency_file_size_and_accuracy_without_column_substitution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            write_matrix(path, list(reversed(matrix_rows())))
            data = load_chart_data(path)
        points = {(p["format"], p["image_size"]): p for p in data["points"]}
        self.assertEqual(len(points), 6)
        self.assertEqual(data["hardware"], "Apple M4 Pro")
        self.assertEqual(data["thread_count"], 8)
        self.assertEqual(data["trained_image_size"], 640)
        self.assertEqual(points[("pytorch_fp32", 640)]["mean_ms"], 10.5)
        self.assertEqual(points[("pytorch_fp32", 640)]["model_size_mb"], 3.0)
        self.assertEqual(points[("pytorch_fp32", 640)]["val_map50_95"], 0.15)
        self.assertEqual(points[("onnx_fp32", 960)]["mean_ms"], 14.5)
        self.assertEqual(points[("onnx_fp32", 960)]["model_size_mb"], 3.4)
        self.assertEqual(points[("onnx_fp32", 960)]["val_map50_95"], 0.19)
        self.assertEqual(points[("onnx_fp32", 960)]["csv_values"]["p95_ms"], "54.5")

    def test_rejects_incomplete_or_mismatched_actual_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            write_matrix(path, matrix_rows()[:-1])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_chart_data(path)
            rows = matrix_rows()
            rows[0]["actual_input_height"] = 384
            write_matrix(path, rows)
            with self.assertRaisesRegex(ValueError, "shape"):
                load_chart_data(path)

    def test_rejects_mixed_hardware_threads_or_wrong_decimal_mb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.csv"
            for changes in ({"hardware": "different CPU"}, {"thread_count": 4},
                            {"model_size_mb": 9.0}):
                with self.subTest(changes=changes):
                    rows = matrix_rows()
                    rows[1].update(changes)
                    write_matrix(path, rows)
                    with self.assertRaises(ValueError):
                        load_chart_data(path)


if __name__ == "__main__":
    unittest.main()
