"""Run the supplied CPU timing loop and append one reproducible PyTorch result."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import uuid

from benchmark_records import append_result, validate_csv
from timing import measure_latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--csv", type=Path, default=Path("results/benchmarks.csv"))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be a positive integer")
    if not args.model.is_file() or not args.image.is_file():
        parser.error("--model and --image must name existing files")
    validate_csv(args.csv)

    root = Path(__file__).resolve().parents[1]
    os.environ["YOLO_CONFIG_DIR"] = str(root / ".cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(root / ".cache/matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    record_directory = args.csv.parent / "benchmark-runs"
    record_directory.mkdir(parents=True, exist_ok=True)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    git_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines()
    hardware = (
        subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Darwin" else platform.processor() or platform.machine()
    )
    started_at = datetime.now(timezone.utc).isoformat()

    mean, p95, fps, times, model, image = measure_latency(str(args.model), str(args.image), args.runs)

    import cv2
    import torch

    library_versions = {"python": platform.python_version()}
    library_versions.update({name: version(name) for name in ("ultralytics", "torch", "opencv-python", "numpy")})
    row = {
        "mean_ms": mean,
        "p95_ms": p95,
        "fps": fps,
        "model_size_mb": args.model.stat().st_size / 1_000_000,
        "image_size": json.dumps(model.predictor.args.imgsz),
        "device": str(model.predictor.device),
        "thread_count": torch.get_num_threads(),
        "library_versions": ";".join(f"{name}={value}" for name, value in library_versions.items()),
        "git_commit": git_commit,
        "hardware": hardware,
        "model_path": str(args.model),
        "image_path": str(args.image),
        "run_count": args.runs,
        "format": "pytorch_fp16" if model.predictor.model.fp16 else "pytorch_fp32",
        "timing_scope": "single-image YOLO predict; logging excluded",
        "p95_method": "sorted samples at int(0.95 * N)",
        "status": "active",
        "superseded_reason": "",
    }
    record = {
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_file": str(args.csv),
        "csv_row": row,
        "summary_line": f"mean {mean:.1f} ms | p95 {p95:.1f} ms | {fps:.1f} FPS",
        "warmup_runs": 10,
        "sorted_samples_ms": times,
        "model_size_bytes": args.model.stat().st_size,
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "original_image_height_width": list(image.shape[:2]),
        "predictor_arguments": vars(model.predictor.args),
        "thread_count_scope": "PyTorch intra-op threads after predictor setup",
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "git_status_at_start": git_status,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("scripts/timing.py", "scripts/benchmark.py", "scripts/benchmark_records.py", "uv.lock")
        },
        "limitations": [
            "Repeated use of one image; not a representative validation-set latency benchmark.",
            "Includes preprocessing, inference and postprocessing; excludes console logging, file loading and warm-up.",
            "image_size is the configured target; rectangular padding can use a smaller tensor.",
            "FPS is the reciprocal of mean call latency, not measured video-pipeline throughput.",
        ],
    }
    record_path = record_directory / f"{uuid.uuid4().hex}.json"
    record_path.write_text(json.dumps(record, indent=2, default=str, allow_nan=False) + "\n")
    append_result(args.csv, row)
    print(f"Saved one CSV row to {args.csv}; run details: {record_path}")


if __name__ == "__main__":
    main()
