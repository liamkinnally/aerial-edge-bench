"""Append benchmark measurements to a consistent CSV schema."""

import csv
from pathlib import Path


CSV_FIELDS = (
    "mean_ms", "p95_ms", "fps", "model_size_mb", "image_size", "device",
    "thread_count", "library_versions", "git_commit", "hardware", "model_path",
    "image_path", "run_count", "format", "timing_scope", "p95_method",
)


def validate_csv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        header = next(csv.reader(handle), None)
    if header != list(CSV_FIELDS):
        raise ValueError(f"CSV header does not match benchmark fields: {path}")


def append_result(path: Path, row: dict) -> None:
    if set(row) != set(CSV_FIELDS):
        raise ValueError("Result must contain exactly the benchmark CSV fields")
    validate_csv(path)
    needs_header = not path.exists() or path.stat().st_size == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
