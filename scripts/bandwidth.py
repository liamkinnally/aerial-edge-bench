"""Compare JPEG-frame payload bandwidth with detection-message payload bandwidth."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import math
import os
from pathlib import Path
import platform
import subprocess


def calculate_bandwidth(model_path, clip_path, FPS=30):
    import cv2, json
    from ultralytics import YOLO
    model = YOLO(model_path)
    video = cv2.VideoCapture(clip_path)
    try:
        if not video.isOpened():
            raise ValueError(f"Clip failed to open: {clip_path}")
        requested_fps = FPS
        clip_fps = video.get(cv2.CAP_PROP_FPS)
        resolution = {"width": int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      "height": int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}
        reported_frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(clip_fps) or clip_fps <= 0:
            raise ValueError("Clip has no valid frame-rate metadata; no measurement performed")
        FPS = clip_fps
        if FPS != requested_fps:
            print(f"Using clip FPS {FPS:g} instead of requested FPS {requested_fps:g}.")

        sizes_a = []   #bytes per frame as a JPEG
        sizes_b = []   # bytes per frame as a detections message
        while True:
            ok, frame = video.read()
            if not ok:
                break
            #this frame as a JPEG
            ok, jpg = cv2.imencode(".jpg", frame)
            sizes_a.append(len(jpg))
            #this frame's detections as a message
            result = model(frame, device="cpu", verbose=False)[0]
            detections = []
            for box in result.boxes:
                detections.append({
                    "cls": int(box.cls),
                    "conf": round(float(box.conf), 2),
                    "box": [round(v) for v in box.xyxy[0].tolist()],
                })
            sizes_b.append(len(json.dumps(detections).encode()))
        avg_a = sum(sizes_a) / len(sizes_a)
        avg_b = sum(sizes_b) / len(sizes_b)
        per_sec_a = avg_a * FPS
        per_sec_b = avg_b * FPS
        ratio = per_sec_a / per_sec_b
        if reported_frame_count > 0 and len(sizes_a) != reported_frame_count:
            raise ValueError("Decoded frame count differs from clip metadata; no result saved")
        print(f"JPEG-frame payload bandwidth {per_sec_a/1000:.0f} kB/s | detections {per_sec_b/1000:.1f} kB/s | {ratio:.0f}x less")

        per_sec_clip = os.path.getsize(clip_path) / (len(sizes_a) / FPS)
        ratio_clip = per_sec_clip / per_sec_b
        print(f"average MP4 file bytes per second {per_sec_clip:.1f} B/s | MP4/detections {ratio_clip:.1f}x")

        return {
            "baseline": "JPEG-frame payload bandwidth",
            "comparison": "Detection-message payload bandwidth",
            "avg_a": avg_a, "avg_b": avg_b,
            "per_sec_a": per_sec_a, "per_sec_b": per_sec_b, "ratio": ratio,
            "clip_baseline": "average MP4 file bytes per second",
            "per_sec_clip": per_sec_clip, "ratio_clip": ratio_clip,
            "clip_file_size_bytes": os.path.getsize(clip_path),
            "clip_duration_seconds": len(sizes_a) / FPS,
            "clip_duration_method": "Decoded frame count divided by recorded clip FPS",
            "clip_summary_line": f"average MP4 file bytes per second {per_sec_clip:.1f} B/s | MP4/detections {ratio_clip:.1f}x",
            "clip_scope": "Average bytes of the whole MP4 file, including container contents; transport overhead is excluded. These are two baselines for this clip, not deployment bounds. Smaller detection messages can also reflect missed detections.",
            "units": {"avg_a": "bytes/frame", "avg_b": "bytes/frame",
                      "per_sec_a": "bytes/second", "per_sec_b": "bytes/second",
                      "ratio": "JPEG payload bytes / detection payload bytes",
                      "per_sec_clip": "bytes/second",
                      "ratio_clip": "MP4 file bytes / detection payload bytes"},
            "sizes_a": sizes_a, "sizes_b": sizes_b,
            "frame_count": len(sizes_a), "reported_frame_count": reported_frame_count,
            "FPS": FPS, "requested_fps": requested_fps,
            "fps_source": "OpenCV VideoCapture CAP_PROP_FPS",
            "resolution": resolution, "capture_backend": video.getBackendName(),
            "jpeg_quality": 95, "jpeg_quality_source": "OpenCV default; no encoder parameters supplied",
            "jpeg_quality_documentation": "https://docs.opencv.org/5.0/main_modules/imgcodecs_flags.html",
            "predictor_arguments": vars(model.predictor.args),
            "summary_line": f"JPEG-frame payload bandwidth {per_sec_a/1000:.0f} kB/s | detections {per_sec_b/1000:.1f} kB/s | {ratio:.0f}x less",
            "scope": "Payload-only estimate at the clip frame rate, not measured network traffic or inference throughput. JPEG frames do not model the original MP4 codec's temporal compression; transport overhead is excluded.",
        }
    finally:
        video.release()


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/phase1-fifty-epochs/weights/best.pt"))
    parser.add_argument("--clip", type=Path, default=Path("data/raw/woodstock-drone-preview.mp4"))
    parser.add_argument("--fps", type=float, default=30,
                        help="Requested FPS; valid clip metadata takes precedence.")
    parser.add_argument("--output", type=Path, default=Path("results/phase4/bandwidth.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Result already exists: {args.output}; choose a new output path")
    if not args.model.is_file() or not args.clip.is_file():
        parser.error("--model and --clip must name existing files")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")

    root = Path(__file__).resolve().parents[1]
    os.environ["YOLO_CONFIG_DIR"] = str(root / ".cache/ultralytics")
    os.environ["MPLCONFIGDIR"] = str(root / ".cache/matplotlib")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    record = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(args.model), "clip_path": str(args.clip),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "clip_sha256": hashlib.sha256(args.clip.read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_status_at_start": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).splitlines(),
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ("scripts/bandwidth.py", "pyproject.toml", "uv.lock")},
        "hardware": (subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                     if platform.system() == "Darwin" else platform.processor() or platform.machine()),
        "platform": platform.platform(), "python": platform.python_version(),
        "library_versions": {name: version(name) for name in ("ultralytics", "torch", "opencv-python", "numpy")},
        "device": "cpu",
    }
    record.update(calculate_bandwidth(str(args.model), str(args.clip), args.fps))
    import cv2
    import torch
    record.update(finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  thread_count=torch.get_num_threads(),
                  thread_count_scope="PyTorch intra-op threads after predictor setup, not total process threads",
                  torch_interop_threads=torch.get_num_interop_threads(),
                  opencv_reported_threads=cv2.getNumThreads())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(record, indent=2, default=str, allow_nan=False) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
