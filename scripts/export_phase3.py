"""Export one static ONNX artifact with an auditable calibration record."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from unittest.mock import patch


def inspect_graph(path, image_size, quantize):
    import onnx
    from onnx import TensorProto

    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    inputs = graph.graph.input
    shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    if len(inputs) != 1 or shape != [1, 3, image_size, image_size]:
        raise ValueError(f"Unexpected ONNX input shape: {shape}")
    if inputs[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Expected FP32 input tensors for both FP32 and INT8 QDQ graphs")
    nodes = Counter(node.op_type for node in graph.graph.node)
    integer_weights = sum(t.data_type in (TensorProto.INT8, TensorProto.UINT8) for t in graph.graph.initializer)
    fp16_weights = sum(t.data_type == TensorProto.FLOAT16 for t in graph.graph.initializer)
    if quantize == 8 and not (integer_weights and nodes["QuantizeLinear"] and nodes["DequantizeLinear"]):
        raise ValueError("INT8 graph lacks integer initializers or Q/DQ nodes")
    if quantize == 32 and (integer_weights or fp16_weights or nodes["QuantizeLinear"] or nodes["DequantizeLinear"]):
        raise ValueError("FP32 graph contains reduced-precision initializers or Q/DQ nodes")
    return {
        "input_shape": shape, "input_dtype": "float32", "node_counts": dict(nodes),
        "int8_initializers": integer_weights, "fp16_initializers": fp16_weights,
        "opsets": {op.domain: op.version for op in graph.opset_import},
        "metadata": {item.key: item.value for item in graph.metadata_props},
    }


def main():
    from phase3_common import ROOT, configure_environment, provenance, sha256, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(640, 960), required=True)
    parser.add_argument("--quantize", type=int, choices=(32, 8), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()
    configure_environment()
    import torch
    from ultralytics import YOLO, settings
    from ultralytics.engine.exporter import Exporter

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    settings.update({"datasets_dir": str(ROOT / "datasets")})
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = provenance()
    metadata.update(status="running", source_weights=str(args.weights), source_weights_sha256=sha256(args.weights))
    configuration = {
        "format": "onnx", "quantize": args.quantize, "imgsz": args.size,
        "dynamic": False, "batch": 1, "device": "cpu", "opset": args.opset,
        "simplify": not args.no_simplify, "nms": None, "max_det": 902,
    }
    if args.quantize == 8:
        configuration.update(data="VisDrone.yaml", split="train", fraction=1.0)
    metadata["configuration"] = configuration
    manifest = args.output / "export.json"
    write_json(manifest, metadata)
    original_loader = Exporter.get_int8_calibration_dataloader

    def record_calibration(exporter, *loader_args, **loader_kwargs):
        loader = original_loader(exporter, *loader_args, **loader_kwargs)
        paths = list(loader.dataset.im_files)
        if not paths or any(Path(path).parent.name != "train" for path in paths):
            raise ValueError("Calibration must contain only VisDrone training images")
        write_json(args.output / "calibration.json", {
            "data": "VisDrone.yaml", "split": "train", "fraction": 1.0,
            "image_count": len(paths), "image_files": paths,
            "input_shape": list(exporter.im.shape), "batch": loader.batch_size,
            "seed": 0, "selection": "Entire native VisDrone training split; no validation images",
            "dataset_manifest": "results/phase1-setup/dataset-manifest.json",
        })
        return loader

    try:
        local_weights = args.output / "best.pt"
        shutil.copyfile(args.weights, local_weights)
        with patch.object(Exporter, "get_int8_calibration_dataloader", record_calibration):
            artifact = Path(YOLO(str(local_weights)).export(**configuration)).resolve()
        metadata["graph"] = inspect_graph(artifact, args.size, args.quantize)
        metadata.update(status="completed", artifact=str(artifact), model_size_bytes=artifact.stat().st_size,
                        model_size_mb=artifact.stat().st_size / 1_000_000, artifact_sha256=sha256(artifact))
    except BaseException as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(manifest, metadata)
    print(json.dumps({key: metadata[key] for key in ("artifact", "model_size_bytes", "artifact_sha256")}))


if __name__ == "__main__":
    main()
