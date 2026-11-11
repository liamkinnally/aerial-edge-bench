import csv
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_records import CSV_FIELDS, append_result, validate_csv


class BenchmarkRecordsTests(unittest.TestCase):
    def test_append_roundtrip_and_single_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "results.csv"
            row = dict.fromkeys(CSV_FIELDS, "value")
            row["hardware"] = 'CPU, "test machine"'
            validate_csv(path)
            self.assertFalse(path.exists())
            append_result(path, row)
            append_result(path, row)
            with path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.reader(handle))
            self.assertEqual(records[0], list(CSV_FIELDS))
            self.assertEqual(len(records), 3)
            self.assertEqual(records[1], [row[key] for key in CSV_FIELDS])
            self.assertEqual(records[2], records[1])

    def test_bad_header_preserves_existing_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            original = b"wrong,header\nkeep,this\n"
            path.write_bytes(original)
            with self.assertRaises(ValueError):
                validate_csv(path)
            with self.assertRaises(ValueError):
                append_result(path, dict.fromkeys(CSV_FIELDS, "value"))
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_row_keys_leave_file_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            valid = dict.fromkeys(CSV_FIELDS, "value")
            missing = valid.copy()
            del missing[CSV_FIELDS[0]]
            extra = {**valid, "unexpected": "value"}
            for row in (missing, extra):
                with self.subTest(keys=list(row)):
                    with self.assertRaises(ValueError):
                        append_result(path, row)
                    self.assertFalse(path.exists())
            append_result(path, valid)
            original = path.read_bytes()
            for row in (missing, extra):
                with self.subTest(existing=True, keys=list(row)):
                    with self.assertRaises(ValueError):
                        append_result(path, row)
                    self.assertEqual(path.read_bytes(), original)

    def test_empty_file_receives_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            path.touch()
            validate_csv(path)
            append_result(path, dict.fromkeys(CSV_FIELDS, "value"))
            with path.open(newline="", encoding="utf-8") as handle:
                records = list(csv.reader(handle))
            self.assertEqual(records[0], list(CSV_FIELDS))
            self.assertEqual(len(records), 2)


if __name__ == "__main__":
    unittest.main()
