"""Run pretrained detection on a local video. This is not a benchmark."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Local drone video")
    parser.add_argument("--weights", type=Path, default=Path("models/yolo26n.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU threads")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--project", type=Path, default=Path("results"))
    parser.add_argument("--name", default="phase0")
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Video does not exist: {args.source}")
    if args.threads < 1:
        parser.error("--threads must be positive")

    # Keep library configuration and caches inside the project when run from its root.
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)

    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))

    def configure_prediction(predictor):
        # Ultralytics resets CPU threads when it selects a device. Apply the
        # requested count after that setup, before warm-up and prediction.
        torch.set_num_threads(args.threads)
        print(
            f"Prediction device={predictor.device}, "
            f"CPU threads={torch.get_num_threads()}, "
            f"interop threads={torch.get_num_interop_threads()}"
        )

    model.add_callback("on_predict_start", configure_prediction)
    # Consuming the generator performs inference and finalizes the saved video.
    for _ in model.predict(
        source=str(args.source),
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        project=str(args.project.resolve()),
        name=args.name,
        stream=True,
        save=True,
        exist_ok=False,
        verbose=False,
    ):
        pass


if __name__ == "__main__":
    main()
