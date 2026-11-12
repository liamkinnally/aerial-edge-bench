# Measurement methodology

[Project overview](../README.md) · [Reproduction commands](REPRODUCE.md) · [Decisions log](../DECISIONS.md)

## Model and validation

All six configurations use one YOLO26 nano checkpoint, fine-tuned on VisDrone for 50 epochs at image size 640, batch 16 and seed 0. Training used MPS. Validation and timing used the Apple M4 Pro CPU.

Validation covers all 548 images in the VisDrone validation split. Metrics come from Ultralytics on converted VisDrone labels, rather than the official challenge evaluation toolkit. They do not measure accuracy on the separate demo clip.

The 960 configurations change inference size only. Training at 960 would be a separate experiment. A fixed training seed does not guarantee identical MPS training results.

## Input shape and runtime controls

Within each input-size group, runtime comparisons use the same checkpoint and matching actual input shapes. Shape guards stop a run if the tensor dimensions differ from the required square input. The shared configuration and shape checks are included in the timed prediction call.

The saved detection head uses external non-maximum suppression. Exports and measurements preserve that mode across formats. NMS-free inference remains a proposed follow-up.

The runs use 8 model intra-op threads. OpenCV's GCD backend did not honor the requested positive thread limit. Its reported setting was consistent across the matrix; the model thread count is not a total-process thread limit.

## Timing

Each row uses 10 warm-up calls followed by 100 predictions of one saved validation image on a quiet laptop. The timed interval includes preprocessing, inference, postprocessing and the common configuration/shape checks. It excludes image-file loading, warm-up and console logging.

This setup compares repeated prediction calls on one image. It does not establish representative latency across the validation set, sustained performance under thermal load, or complete video-pipeline throughput. The timing loop's reciprocal-latency FPS value is not a measurement of a running video pipeline.

The p95 follows the supplied loop's sorted-sample indexing convention: the zero-based sample at `int(0.95 * N)`. It describes the recorded samples and is not a worst-case guarantee.

The 22.2 ms mean for ONNX FP32 at 960 and 22.9 ms for PyTorch FP32 at 640 are observed results on the Apple M4 Pro CPU. They do not establish statistical equivalence, equal tail latency, or performance on another processor.

### Historical comparison

An earlier PyTorch timing measurement used a 384 × 640 input; the final matrix uses square inputs. The checkpoint and measurement wrapper also changed between those measurements, so the historical comparison does not isolate the cost of padding. See the [decisions log](../DECISIONS.md) for the original comparison.

## Precision and file size

File sizes use decimal MB. Quantization's size tradeoff compares ONNX INT8 with ONNX FP32 at the same input size.

The PyTorch checkpoint is stored differently from an ONNX graph. Its FP32 label describes execution precision, so its file size is not an equal-format compression baseline. INT8 uses quantized weighted operations with float input/output and some remaining float operations.

At input size 640, INT8 reduced the ONNX file size by 69.1%, while mAP50–95 fell from 0.169 to 0.153. INT8 did not reduce mean latency on the Apple M4 Pro CPU in either input-size group.

## Payload calculations

The same fine-tuned PyTorch model processed 353 frames from a 640 × 1138 preview clip on the Apple M4 Pro CPU. The clip's recorded rate of 24 frames/s scales the calculated payload rates; it is not a measurement of processing throughput. Rates use decimal kB/s.

- **Detection messages:** the script's JSON list of classes, confidence scores and pixel-coordinate boxes, including an empty list when nothing is detected.
- **MP4 baseline:** the whole clip file size divided by decoded-frame duration, including container contents. Duration is the decoded frame count divided by recorded FPS.
- **JPEG baseline:** each frame encoded independently using OpenCV's default quality of 95.

These are two named baselines for one clip, not bounds on a deployed link. They exclude transport overhead and are not measured network traffic. Detection messages retain selected information from the images rather than reproducing their visual content.

There were 78 empty detection messages among 353 frames; the largest message was 553 bytes. A small message can reflect missed objects, so the payload comparison should be read alongside the detection accuracy results. Accuracy on the demo clip itself was not measured.

The [saved summary](measurements/summary.json) contains the underlying values. The MP4 comparison was calculated from the original measurement without rerunning inference.

## Qualitative comparison

The before/after image compares stock COCO weights with the final VisDrone checkpoint on the same saved [source frame](assets/source-frame.jpg). COCO and VisDrone have different class vocabularies. Additional boxes alone do not prove better accuracy; the illustration is separate from validation.

Footage is credited to Jordan Koschei / ConnectHV under CC0. Provenance is in [data/SOURCES.md](../data/SOURCES.md).

## Deployment scope and reproduction

No target drone computer or radio link was tested. Heat, power, memory under sustained inference, deployment latency and actual camera/link behavior remain unmeasured.

The [reproduction guide](REPRODUCE.md) provides commands and environment requirements. The repository contains selected measurement summaries and images. Model weights, datasets and full original run logs remain local and ignored; the guide identifies those original records. Library versions, configuration and artifact identifiers are recorded in [summary.json](measurements/summary.json).
