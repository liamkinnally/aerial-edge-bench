"""Append benchmark measurements to a consistent CSV schema."""

import csv
from pathlib import Path
import tempfile


CSV_FIELDS = (
    "mean_ms", "p95_ms", "fps", "model_size_mb", "image_size", "device",
    "thread_count", "library_versions", "git_commit", "hardware", "model_path",
    "image_path", "run_count", "format", "timing_scope", "p95_method",
    "status", "superseded_reason",
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


def _read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or any(len(row) != len(rows[0]) for row in rows[1:]):
        raise ValueError(f"CSV has missing or extra cells: {path}")
    return rows


def _replace_rows(path: Path, rows: list[list[str]]) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            csv.writer(handle).writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def migrate_legacy_csv(path: Path) -> None:
    """Add status fields to the exact original schema without changing old cells."""
    rows = _read_rows(path)
    if rows[0] == list(CSV_FIELDS):
        return
    if rows[0] != list(CSV_FIELDS[:-2]):
        raise ValueError(f"CSV header is not a recognized benchmark schema: {path}")
    _replace_rows(path, [list(CSV_FIELDS)] + [row + ["active", ""] for row in rows[1:]])


def supersede_result(path: Path, row_index: int, reason: str) -> None:
    """Mark one active data row superseded after its replacement has been saved."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("A nonempty superseded reason is required")
    rows = _read_rows(path)
    if rows[0] != list(CSV_FIELDS):
        raise ValueError(f"CSV header does not match benchmark fields: {path}")
    if type(row_index) is not int or not 0 <= row_index < len(rows) - 1:
        raise ValueError("Superseded row index is out of range")
    row = rows[row_index + 1]
    status_index = CSV_FIELDS.index("status")
    if row[status_index] != "active":
        raise ValueError("Only an active row can be superseded")
    row[status_index] = "superseded"
    row[CSV_FIELDS.index("superseded_reason")] = reason
    _replace_rows(path, rows)
