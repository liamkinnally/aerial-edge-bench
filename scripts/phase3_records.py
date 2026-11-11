"""Validate the fixed Phase 3 experiment matrix and retain native accuracy values."""

import csv
import math
from pathlib import Path
import tempfile


CSV_FIELDS = (
    "format", "image_size", "actual_input_height", "actual_input_width",
    "val_map50_95", "val_map50", "pedestrian_ap50_95", "pedestrian_ap50",
    "mean_ms", "p95_ms", "fps", "model_size_bytes", "model_size_mb", "device",
    "thread_count", "library_versions", "git_commit", "hardware", "model_path",
    "image_path", "run_count", "trained_image_size", "size_experiment", "run_record",
)
FORMATS = frozenset({"pytorch_fp32", "onnx_fp32", "onnx_int8"})
SIZES = frozenset({640, 960})
SIZE_EXPERIMENT = "inference size only; trained at 640"
ACCURACY_FIELDS = ("val_map50_95", "val_map50", "pedestrian_ap50_95", "pedestrian_ap50")


def _number(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if isinstance(value, bool) or not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _score(value, name: str) -> float:
    score = _number(value, name)
    if not 0 <= score <= 1:
        raise ValueError(f"{name} must be between zero and one")
    return score


def _positive_integer(value, name: str) -> int:
    number = _number(value, name)
    if number <= 0 or not number.is_integer():
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def extract_accuracy(aggregate: dict, per_class: list) -> dict:
    """Extract native aggregate and pedestrian AP without relying on class order."""
    if not isinstance(aggregate, dict) or not isinstance(per_class, list):
        raise ValueError("Expected native aggregate dictionary and per-class list")
    if any(not isinstance(row, dict) for row in per_class):
        raise ValueError("Per-class records must be dictionaries")
    pedestrian = [row for row in per_class if row.get("Class") == "pedestrian"]
    if len(pedestrian) != 1:
        raise ValueError("Expected exactly one pedestrian per-class record")
    try:
        values = {
            "val_map50_95": aggregate["metrics/mAP50-95(B)"],
            "val_map50": aggregate["metrics/mAP50(B)"],
            "pedestrian_ap50_95": pedestrian[0]["mAP50-95"],
            "pedestrian_ap50": pedestrian[0]["mAP50"],
        }
    except KeyError as error:
        raise ValueError(f"Missing native accuracy field: {error.args[0]}") from error
    return {name: _score(value, name) for name, value in values.items()}


def _validate_rows(rows: list[dict]) -> set[tuple[str, int]]:
    seen = set()
    shapes = {}
    for row in rows:
        if set(row) != set(CSV_FIELDS):
            raise ValueError("Result must contain exactly the Phase 3 CSV fields")
        size = _positive_integer(row["image_size"], "image_size")
        if row["format"] not in FORMATS or size not in SIZES:
            raise ValueError("Unexpected Phase 3 format or image size")
        height = _positive_integer(row["actual_input_height"], "actual_input_height")
        width = _positive_integer(row["actual_input_width"], "actual_input_width")
        shape = (height, width)
        if shape != (size, size) or (size in shapes and shapes[size] != shape):
            raise ValueError("Actual input shape differs from the required shape for this size")
        shapes[size] = shape
        key = (row["format"], size)
        if key in seen:
            raise ValueError(f"Duplicate Phase 3 format/size row: {key}")
        seen.add(key)
        if row["device"] != "cpu":
            raise ValueError("Phase 3 measurements must use CPU")
        if _positive_integer(row["trained_image_size"], "trained_image_size") != 640:
            raise ValueError("The model must be recorded as trained at 640")
        if row["size_experiment"] != SIZE_EXPERIMENT:
            raise ValueError("The inference-size-only limitation must be recorded")
        for name in ACCURACY_FIELDS:
            _score(row[name], name)
        for name in ("model_size_bytes", "thread_count", "run_count"):
            _positive_integer(row[name], name)
        for name in ("mean_ms", "p95_ms", "fps", "model_size_mb"):
            if _number(row[name], name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("library_versions", "git_commit", "hardware", "model_path", "image_path", "run_record"):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError(f"{name} must be recorded")
    return seen


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle, strict=True)
            if next(reader, None) != list(CSV_FIELDS):
                raise ValueError(f"CSV header does not match the Phase 3 fields: {path}")
            rows = []
            for cells in reader:
                if len(cells) != len(CSV_FIELDS):
                    raise ValueError(f"CSV row has missing or extra cells: {path}")
                rows.append(dict(zip(CSV_FIELDS, cells)))
            return rows
    except csv.Error as error:
        raise ValueError(f"Malformed Phase 3 CSV: {path}") from error


def append_matrix_row(path: Path, row: dict) -> None:
    """Accept one valid new combination, or leave the existing CSV untouched."""
    path = Path(path)
    rows = [*_read_rows(path), row]
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def complete_matrix(path: Path) -> list[dict]:
    """Return the six recorded rows only when the complete matrix is valid."""
    rows = _read_rows(Path(path))
    combinations = _validate_rows(rows)
    if combinations != {(model_format, size) for model_format in FORMATS for size in SIZES}:
        raise ValueError("Phase 3 matrix is incomplete; all six format/size rows are required")
    return rows
