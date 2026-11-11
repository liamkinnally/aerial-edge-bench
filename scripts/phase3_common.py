"""Environment and provenance shared by the export and matrix commands."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def configure_environment():
    os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ.pop("ORT_INTRA_OP_NUM_THREADS", None)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def provenance():
    sources = ["scripts/timing.py", "scripts/phase3_common.py", "scripts/export_phase3.py",
               "scripts/phase3_runtime.py", "scripts/phase3_records.py", "scripts/benchmark_phase3.py",
               "pyproject.toml", "uv.lock"]
    return {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "library_versions": {name: version(name) for name in (
            "ultralytics", "torch", "torchvision", "opencv-python", "numpy", "onnx", "onnxruntime", "onnxslim")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines(),
        "source_sha256": {name: sha256(ROOT / name) for name in sources},
    }
