"""Fine-tune on VisDrone with MPS and retain native validation outputs."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from training_records import EpochRecorder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, choices=(1, 10), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["YOLO_CONFIG_DIR"] = str(root / ".cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(root / ".cache/matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

    import torch
    from ultralytics import YOLO, settings

    if not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable; stopping without CPU training.")
    settings.update({"datasets_dir": str(root / "datasets")})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    weights = root / "models/yolo26n.pt"
    configuration = {
        "data": "VisDrone.yaml", "epochs": args.epochs, "imgsz": 640,
        "device": "mps", "batch": 8, "workers": 0, "cache": False,
        "seed": 0, "deterministic": True, "amp": False, "close_mosaic": 0,
        "val": True, "save": True, "plots": True, "pretrained": True,
        "project": str(output.parent), "name": output.name, "exist_ok": True,
    }
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "purpose": "VisDrone fine-tuning; no inference benchmark or export",
        "configuration": configuration,
        "device": "mps",
        "mps_available": torch.backends.mps.is_available(),
        "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
        "physical_memory_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": {d.metadata["Name"]: d.version for d in distributions()},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines(),
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("scripts/train.py", "scripts/training_records.py", "pyproject.toml", "uv.lock")
        },
        "initial_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "dataset_manifest": "results/phase1-setup/dataset-manifest.json",
        "continuation_policy": "Each invocation starts from the same pretrained weights; pilot is separate from ten-epoch run.",
    }
    manifest = output / "run-metadata.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    recorder = EpochRecorder(
        output,
        synchronize=torch.mps.synchronize,
        memory_usage=lambda: {
            "mps_tensor_bytes": torch.mps.current_allocated_memory(),
            "mps_driver_bytes": torch.mps.driver_allocated_memory(),
        },
    )
    started = time.perf_counter()
    try:
        model = YOLO(str(weights))
        model.add_callback("on_train_epoch_start", recorder.on_epoch_start)
        model.add_callback("on_fit_epoch_end", recorder.on_fit_epoch_end)
        model.train(**configuration)
        if len(recorder.epochs) != args.epochs:
            raise RuntimeError(f"Expected {args.epochs} complete epochs, recorded {len(recorder.epochs)}")
        metadata["status"] = "completed"
        metadata["actual_training_device"] = str(model.trainer.device)
        metadata["actual_validation_device"] = str(model.trainer.validator.device)
        metadata["actual_torch_threads"] = torch.get_num_threads()
        metadata["saved_weights"] = {
            path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in (model.trainer.best, model.trainer.last)
        }
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            recorder.close()
        finally:
            metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["training_call_wall_seconds"] = time.perf_counter() - started
            metadata["training_call_scope"] = "model.train including setup and final best-checkpoint validation; epoch time is in epoch-records.json"
            manifest.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
