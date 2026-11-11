import csv
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_records import (
    CSV_FIELDS, append_result, migrate_legacy_csv, supersede_result, validate_csv,
)


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

    def test_legacy_migration_preserves_cells_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            legacy_fields = list(CSV_FIELDS[:-2])
            cells = [f"original-{field}" for field in legacy_fields]
            cells[0] = "16.6863196156919000"
            cells[legacy_fields.index("hardware")] = 'CPU, "quoted"'
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle).writerows([legacy_fields, cells])
            migrate_legacy_csv(path)
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
            self.assertEqual(rows, [list(CSV_FIELDS), cells + ["active", ""]])
            migrated = path.read_bytes()
            migrate_legacy_csv(path)
            self.assertEqual(path.read_bytes(), migrated)

    def test_bad_legacy_header_or_row_width_leaves_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            for rows in (
                [["unexpected", "header"], ["keep", "this"]],
                [list(CSV_FIELDS[:-2]), ["short-row"]],
            ):
                with self.subTest(rows=rows):
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle).writerows(rows)
                    original = path.read_bytes()
                    with self.assertRaises(ValueError):
                        migrate_legacy_csv(path)
                    self.assertEqual(path.read_bytes(), original)

    def test_supersede_updates_only_selected_row(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            first = dict.fromkeys(CSV_FIELDS, "original")
            first.update(status="active", superseded_reason="")
            second = {**first, "mean_ms": "new-value"}
            append_result(path, first)
            append_result(path, second)
            supersede_result(path, 0, 'Logging included; replaced by "quiet" run')
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [
                {**first, "status": "superseded", "superseded_reason": 'Logging included; replaced by "quiet" run'},
                second,
            ])
            original = path.read_bytes()
            with self.assertRaises(ValueError):
                supersede_result(path, 0, "Another reason")
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_supersede_request_leaves_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            row = dict.fromkeys(CSV_FIELDS, "original")
            row.update(status="active", superseded_reason="")
            append_result(path, row)
            original = path.read_bytes()
            for index, reason in ((-1, "reason"), (1, "reason"), (0, ""), (0, "   ")):
                with self.subTest(index=index, reason=reason):
                    with self.assertRaises(ValueError):
                        supersede_result(path, index, reason)
                    self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
