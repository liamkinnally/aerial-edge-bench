"""Validate and time one Phase 3 model at a checked square CPU input shape."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

from phase3_common import configure_environment, provenance, sha256, write_json
from phase3_records import FORMATS, SIZE_EXPERIMENT, append_matrix_row, extract_accuracy
from timing import measure_latency


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", choices=sorted(FORMATS), required=True)
    parser.add_argument("--size", type=int, choices=(640, 960), required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("results/phase3/benchmarks.csv"))
    args = parser.parse_args()
    configure_environment()
    import cv2
    import psutil
    import torch
    from ultralytics import YOLO, settings
    from phase3_runtime import make_guarded_types
    from phase3_common import ROOT

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    settings.update({"datasets_dir": str(ROOT / "datasets")})
    args.output.mkdir(parents=True, exist_ok=False)
    record = provenance()
    record.update(status="running", model_path=str(args.model), model_sha256=sha256(args.model),
                  image_path=str(args.image), image_sha256=sha256(args.image),
                  source_weights="results/phase1-fifty-epochs/weights/best.pt",
                  source_weights_sha256=sha256(ROOT / "results/phase1-fifty-epochs/weights/best.pt"),
                  trained_image_size=640, size_experiment=SIZE_EXPERIMENT, seed=0)
    record_path = args.output / "run.json"
    write_json(record_path, record)
    try:
        if args.format == "pytorch_fp32":
            if args.model.suffix != ".pt" or record["model_sha256"] != record["source_weights_sha256"]:
                raise ValueError("PyTorch row must use the unchanged fifty-epoch best checkpoint")
        else:
            from export_phase3 import inspect_graph
            record["graph"] = inspect_graph(args.model, args.size, 8 if args.format == "onnx_int8" else 32)
            exported = json.loads((args.model.parent / "export.json").read_text())
            if exported["status"] != "completed" or exported["source_weights_sha256"] != record["source_weights_sha256"]:
                raise ValueError("ONNX provenance does not match the fifty-epoch checkpoint")
            if exported["artifact_sha256"] != record["model_sha256"]:
                raise ValueError("ONNX artifact hash differs from its successful export record")
        ConfiguredYOLO, GuardedValidator = make_guarded_types(args.size)
        validation_arguments = {
            "data": "VisDrone.yaml", "split": "val", "imgsz": args.size,
            "batch": 1, "rect": False, "device": "cpu", "workers": 0,
            "quantize": 32, "conf": 0.001, "iou": 0.7, "max_det": 902,
            "nms": None, "plots": False, "verbose": False, "seed": 0,
            "project": str(args.output.resolve()), "name": "validation", "exist_ok": True,
        }
        record["validation_arguments"] = validation_arguments
        metrics = YOLO(str(args.model), task="detect").val(validator=GuardedValidator, **validation_arguments)
        validator = GuardedValidator.last_instance
        record["validation_shapes"] = validator.shape_audit.as_dict()
        record["validation_images"] = validator.seen
        record["validation_threads"] = validator.torch_threads
        record["validation_actual_arguments"] = vars(validator.args)
        if validator.seen != 548 or validator.shape_audit.count != validator.seen:
            raise ValueError("Validation did not cover the complete recorded VisDrone val split at batch one")
        write_json(args.output / "validation-metrics.json", {key: float(value) for key, value in metrics.results_dict.items()})
        (args.output / "validation-per-class.json").write_text(metrics.to_json(decimals=12) + "\n")
        accuracy = extract_accuracy(
            json.loads((args.output / "validation-metrics.json").read_text()),
            json.loads((args.output / "validation-per-class.json").read_text()),
        )
        record["validation_status"] = "completed"
        write_json(record_path, record)

        samples = []
        for _ in range(60):
            samples.append(psutil.cpu_percent(interval=1.0))
            if len(samples) >= 3 and all(value <= 5.0 for value in samples[-3:]):
                break
        quiet = {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
                 "system_cpu_percent": samples, "threshold_percent": 5.0,
                 "passed": len(samples) >= 3 and all(value <= 5.0 for value in samples[-3:]),
                 "scope": "Three consecutive pre-run one-second CPU observations; no exclusive-machine guarantee"}
        write_json(args.output / "quiet-machine.json", quiet)
        if not quiet["passed"]:
            raise RuntimeError("Machine did not become quiet; no timing loop was invoked")
        record["timing_started_at_utc"] = datetime.now(timezone.utc).isoformat()
        # The supplied function imports YOLO internally. This scoped binding only
        # supplies the common prediction settings and shape guard; its body is unchanged.
        with patch("ultralytics.YOLO", ConfiguredYOLO):
            mean, p95, fps, times, model, image = measure_latency(str(args.model), str(args.image), 100)
        predictor = model.predictor
        record["timing_shapes"] = predictor.shape_audit.as_dict()
        if predictor.shape_audit.count != 110:
            raise ValueError("Expected ten warm-up and one hundred timed calls")
        backend = predictor.model
        if str(backend.device) != "cpu" or backend.fp16 or torch.get_num_threads() != 8:
            raise ValueError("Unexpected timing device, input precision or thread count")
        if args.format.startswith("onnx"):
            options = backend.session.get_session_options()
            record["onnxruntime"] = {"providers": backend.session.get_providers(),
                                     "intra_op_threads": options.intra_op_num_threads,
                                     "inter_op_threads": options.inter_op_num_threads,
                                     "execution_mode": str(options.execution_mode)}
            if options.intra_op_num_threads != 8 or backend.session.get_providers() != ["CPUExecutionProvider"]:
                raise ValueError("ONNX Runtime must use eight intra-op threads and CPU only")
        record.update(sorted_samples_ms=times, warmup_runs=10, timed_runs=100,
                      summary_line=f"mean {mean:.1f} ms | p95 {p95:.1f} ms | {fps:.1f} FPS",
                      predictor_arguments=vars(predictor.args),
                      torch_intra_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                      opencv_threads=cv2.getNumThreads(), original_image_height_width=list(image.shape[:2]),
                      timing_scope="CPU predict including preprocessing, postprocessing, configuration adapter and actual-shape guard; logging, file loading and warm-up excluded",
                      p95_method="sorted samples at int(0.95 * N)",
                      limitations=["Repeated single-image latency; FPS is reciprocal mean latency, not video throughput",
                                   "INT8 uses QDQ weighted operations with FP32 I/O and remaining float operations",
                                   "Native Ultralytics validation on converted VisDrone labels, not official challenge toolkit scores",
                                   SIZE_EXPERIMENT])
        row = {"format": args.format, "image_size": args.size,
               "actual_input_height": predictor.shape_audit.as_dict()["observed_shapes"][0][2],
               "actual_input_width": predictor.shape_audit.as_dict()["observed_shapes"][0][3],
               **accuracy, "mean_ms": mean, "p95_ms": p95, "fps": fps,
               "model_size_bytes": args.model.stat().st_size, "model_size_mb": args.model.stat().st_size / 1_000_000,
               "device": "cpu", "thread_count": torch.get_num_threads(),
               "library_versions": ";".join([f"python={record['python']}", *[f"{k}={v}" for k, v in record["library_versions"].items()]]),
               "git_commit": record["git_commit"], "hardware": record["hardware"],
               "model_path": str(args.model), "image_path": str(args.image), "run_count": 100,
               "trained_image_size": 640, "size_experiment": SIZE_EXPERIMENT, "run_record": str(record_path)}
        record["csv_row"] = row
        write_json(record_path, record)
        append_matrix_row(args.csv, row)
        record["status"] = "completed"
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(record_path, record)
    print(f"Saved one Phase 3 row to {args.csv}; native evidence: {args.output}")


if __name__ == "__main__":
    main()
