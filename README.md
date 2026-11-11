# Aerial edge benchmark

**ONNX FP32 at 960 × 960 achieved higher accuracy with similar mean prediction latency to PyTorch FP32 at 640 × 640 on the Apple M4 Pro CPU: 22.2 ms versus 22.9 ms.** VisDrone validation mAP50–95 was 0.213 versus 0.169. This is an observed comparison on this machine, not a test of statistical equivalence.

![Stock COCO model on the left and VisDrone fine-tuned model on the right, on the same saved drone frame.](docs/assets/before-after.png)

The left panel uses the stock model; the right uses the final VisDrone checkpoint. This is a qualitative illustration on the same [source frame](docs/assets/source-frame.jpg), separate from validation. COCO and VisDrone have different class vocabularies, and additional boxes alone do not prove better accuracy. Footage: Jordan Koschei / ConnectHV, CC0; provenance is in [data/SOURCES.md](data/SOURCES.md).

[Measured results](docs/measurements/benchmarks.csv) · [Method and bandwidth data](docs/measurements/summary.json) · [Decisions](DECISIONS.md)

## What this is

A laptop experiment on aerial object detection: fine-tune a small detector, compare PyTorch and ONNX execution, quantize the ONNX model, and measure the payload needed to send detection messages instead of images. The laptop CPU is a stand-in for comparing options; target-device performance has not been measured.

All models use the same YOLO26 nano checkpoint, fine-tuned on VisDrone for 50 epochs at image size 640, batch 16 and seed 0. Training used MPS; validation and timing used the Apple M4 Pro CPU. The 960 rows change inference size only.

## Results

Accuracy uses the full VisDrone validation split (548 images). Timing uses 100 repeated predictions of one validation image after 10 warm-up calls, with 8 model intra-op threads on the Apple M4 Pro CPU. The timed interval includes preprocessing, inference, postprocessing and the common configuration/shape checks; it excludes image-file loading, warm-up and console logging.

| Format | Actual input H × W | mAP50–95 | mAP50 | Pedestrian AP50–95 | Mean ms (Apple M4 Pro CPU) | p95 ms (Apple M4 Pro CPU) | File MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch FP32 | 640 × 640 | 0.169 | 0.308 | 0.137 | 22.9 | 24.6 | 5.444 |
| ONNX FP32 | 640 × 640 | 0.169 | 0.308 | 0.137 | 10.9 | 11.6 | 9.861 |
| ONNX INT8 | 640 × 640 | 0.153 | 0.287 | 0.119 | 12.2 | 13.2 | 3.045 |
| PyTorch FP32 | 960 × 960 | 0.213 | 0.375 | 0.199 | 38.5 | 39.1 | 5.444 |
| ONNX FP32 | 960 × 960 | 0.213 | 0.375 | 0.199 | 22.2 | 25.4 | 10.071 |
| ONNX INT8 | 960 × 960 | 0.194 | 0.348 | 0.171 | 23.0 | 23.9 | 3.255 |

AP values are fractions; higher is better. Latency is the full prediction call on the Apple M4 Pro CPU; lower is better. File sizes use decimal MB. Actual input tensors are identical across formats within each input-size group. Shape guards stop a run if those inputs differ.

At input size 640, INT8 reduced the ONNX file size by 69.1% relative to ONNX FP32, while mAP50–95 fell from 0.169 to 0.153. INT8 did not reduce mean latency on the Apple M4 Pro CPU in either input-size group.

The PyTorch checkpoint is stored differently from an ONNX graph. The FP32 label describes its execution precision, so its file size is not an equal-format compression baseline. INT8 uses quantized weighted operations with float input/output and some remaining float operations.

## Speed versus accuracy

![Mean full-call latency on the Apple M4 Pro CPU versus VisDrone validation mAP50–95, with every format and input size labelled.](docs/assets/speed-vs-accuracy.png)

The useful comparison is ONNX FP32 at the larger input size against PyTorch FP32 at the smaller input size. Mean latency is similar on the Apple M4 Pro CPU, while the larger-input result has higher validation accuracy. This does not establish equal tail latency or performance on another processor.

## File size versus accuracy

![Serialized model size in decimal MB versus VisDrone validation mAP50–95, with every format and input size labelled.](docs/assets/file-size-vs-accuracy.png)

Compare ONNX INT8 with ONNX FP32 to assess quantization's size tradeoff. The smaller files retained much of the measured accuracy, but they did not improve mean latency on the Apple M4 Pro CPU in this experiment.

## Bandwidth comparison

The same fine-tuned PyTorch model processed 353 frames from a 640 × 1138 preview clip at its recorded rate of 24 frames/s. The model ran on the Apple M4 Pro CPU. JPEG encoding used the OpenCV default quality of 95.

| Payload | Average bytes/frame | Average kB/s | Baseline ÷ detection payload |
|---|---:|---:|---:|
| Detection messages | 166.5 | 4.0 | — |
| Average MP4 file bytes per second | — | 63.3 | 15.8× |
| JPEG-frame payload bandwidth | 219896.0 | 5,277.5 | 1320.8× |

Rates use decimal kB/s. The MP4 baseline is the whole clip file size divided by decoded-frame duration, including container contents. The JPEG baseline encodes each frame independently. The detection payload is the script's JSON list of classes, confidence scores and pixel-coordinate boxes, including an empty list when nothing is detected.

These are two named baselines for one clip, not bounds on a deployed link. They exclude transport overhead and are not measured network traffic. The frame rate scales payload size; it is not a measurement of processing throughput.

There were 78 empty detection messages among 353 frames; the largest message was 553 bytes. A small message can reflect missed objects. Read the payload comparison together with the detection accuracy results.

The [saved summary](docs/measurements/summary.json) contains the underlying values. The MP4 comparison was calculated from the original measurement without rerunning inference.

## Limitations

- No target drone computer or radio link was tested. Heat, power, memory under sustained inference and deployment latency remain unmeasured.
- The validation metrics come from Ultralytics on converted VisDrone labels, rather than the official challenge evaluation toolkit. They do not measure accuracy on the separate demo clip.
- Timing repeats one image on a quiet laptop. It does not establish video-pipeline throughput or representative latency across the validation set. The p95 follows the supplied loop's sorted-sample indexing convention; it is not a worst-case guarantee.
- OpenCV's GCD backend did not honor the requested positive thread limit. Its reported setting was consistent across the matrix; the model thread count is not a total-process thread limit.
- The saved detection head uses external non-maximum suppression. Exports and measurements preserve that mode across formats.
- Training and inference at a larger image size are different experiments. Only inference size changed here. A fixed training seed also does not guarantee identical MPS training results.
- The historical timing comparison changed the checkpoint and measurement wrapper as well as input shape, so it does not isolate the cost of padding.

Next experiments include a rerun in YOLO26's NMS-free mode, alongside the follow-ups in [DECISIONS.md](DECISIONS.md): training at the larger size, trying a larger model, testing the intended runtime on target hardware, and measuring the actual camera and link. Those are proposed follow-ups.

[How to reproduce](docs/REPRODUCE.md)

## Who wrote what

I wrote the timing loop with a proposed skeleton and review, the original bandwidth math, and the decisions log. Claude helped with planning and supplied the bandwidth frame loop and message format.

Codex built the wrappers, result recording, MP4 calculation, training/export/validation scripts, tests, charts and README. I approved the corrections to the decisions log.

## Licensing

This repository's code is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0); see [LICENSE](LICENSE).

Ultralytics is also distributed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) and offers a separate [Enterprise License](https://www.ultralytics.com/license). The repository's code license does not change the separate dataset and footage terms below.

VisDrone is a research dataset. As recorded in [data/SOURCES.md](data/SOURCES.md), the linked upstream README did not include a standalone license grant. Dataset images, annotations and model weights are not included in this repository. Review the upstream terms before reuse or redistribution.

The separate ConnectHV demo footage is CC0, as documented in [data/SOURCES.md](data/SOURCES.md). That statement does not apply to VisDrone.
