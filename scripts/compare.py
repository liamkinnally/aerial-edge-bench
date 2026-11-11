"""Save stock and fine-tuned predictions on one unchanged source frame."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ["YOLO_CONFIG_DIR"] = str(root / ".cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(root / ".cache/matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"

    import cv2
    import numpy as np
    from ultralytics import YOLO

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Image failed to load: {args.image}")
    args.output.mkdir(parents=True, exist_ok=False)
    settings = dict(device="cpu", imgsz=640, conf=0.25, iou=0.7,
                    max_det=300, rect=True, verbose=False, save=False)
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_frame": str(args.image),
        "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
        "source_height_width": list(image.shape[:2]),
        "prediction_settings": settings,
        "library_versions": {name: version(name) for name in ("ultralytics", "torch", "opencv-python", "numpy")},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines(),
        "source_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Qualitative comparison on the same saved Phase 0 source frame; COCO and VisDrone class vocabularies differ; not an accuracy evaluation.",
        "models": {},
    }
    panels = []
    for name, weights, title in (
        ("before", args.before, "Before: stock COCO model"),
        ("after", args.after, "After: VisDrone fine-tune"),
    ):
        result, = YOLO(str(weights))(image.copy(), **settings)
        predictions = json.loads(result.to_json())
        (args.output / f"{name}-predictions.json").write_text(json.dumps(predictions, indent=2) + "\n")
        record["models"][name] = {
            "weights": str(weights),
            "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "class_names": result.names,
            "detection_count": len(predictions),
            "original_height_width": list(result.orig_shape),
        }
        annotated = result.plot()
        banner = np.full((64, annotated.shape[1], 3), 21, dtype=np.uint8)
        cv2.putText(banner, title, (16, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (223, 228, 228), 2, cv2.LINE_AA)
        panels.append(np.concatenate((banner, annotated), axis=0))
    comparison = np.concatenate(panels, axis=1)
    image_path = args.output / "before-after.png"
    if not cv2.imwrite(str(image_path), comparison):
        raise OSError(f"Failed to write {image_path}")
    record["comparison_height_width"] = list(comparison.shape[:2])
    record["comparison_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
    (args.output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Saved {image_path}")


if __name__ == "__main__":
    main()
