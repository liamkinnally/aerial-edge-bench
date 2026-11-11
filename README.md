# Aerial edge bench

Experiments on the tradeoffs between aerial object detection, model size, CPU inference, and sending detections instead of video.

## Current scope

The initial setup runs a pretrained YOLO26 nano detector on a local drone video and saves an annotated copy. The MPS fine-tuning pilot and short VisDrone training run are also complete, with weights and native validation outputs retained locally under `results/phase1-pilot/` and `results/phase1-ten-epochs/`. A CPU timing command measures repeated predictions on one image. Quantization, export, and bandwidth analysis have not run.

The pretrained model uses COCO classes. Detection errors on small aerial objects are expected; this demo does not establish VisDrone accuracy or performance on a drone computer. The later CPU experiments will use a laptop CPU as a stand-in.

## Run the demo

Install [uv](https://docs.astral.sh/uv/), then run these commands from the repository root. The Python version is in `.python-version`; `uv.lock` pins the complete dependency environment.

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --locked
mkdir -p data/raw
curl -fL --retry 2 \
  'https://media.connecthv.com/preview/Flying%20through%20downtown%20Woodstock.mp4' \
  -o data/raw/woodstock-drone-preview.mp4
uv run --locked python scripts/predict.py \
  --source data/raw/woodstock-drone-preview.mp4
```

The first prediction downloads the official pretrained weights into `models/`. The annotated video is saved under `results/phase0/`; repeat runs receive a new directory. These files stay out of Git. The wrapper exposes the device, thread count, image size, and confidence threshold through `--help`. Its library console timings are diagnostic output, not benchmark results.

After downloading the weights, run the checks with `uv run --locked python -m unittest discover -s tests -v`. They cover prediction output, CPU thread configuration, epoch records, and failure metadata.

## Fine-tuning

Training requires MPS and stops if it is unavailable. Run a pilot from the repository root, using a new output directory:

```sh
uv run --locked python scripts/train.py --epochs 1 --output results/training-pilot
```

The installed `VisDrone.yaml` downloads and converts the dataset under the ignored `datasets/` directory when needed. Each invocation starts from the original pretrained weights. Check the pilot's measured time and memory before choosing the longer run.

The longer run uses an explicit batch size and a new output directory:

```sh
uv run --locked python scripts/train.py --epochs 50 --batch 16 --output results/phase1-fifty-epochs
```

If the library attempts to reduce the batch after a memory failure, the wrapper stops and records the attempted change. Preserve that failed directory before starting a fresh run with the agreed smaller batch in another directory.

The training directory contains `weights/best.pt`, `weights/last.pt`, the native `results.csv`, per-epoch validation JSON, and separate final-best aggregate and per-class validation JSON. `run-metadata.json` records the seed, device, package versions, Git revision, and source hashes. `epoch-records.json` includes validation and checkpoint saving in each epoch's wall time; it labels sampled MPS memory separately from process peak resident memory.

These are native Ultralytics validation results on converted VisDrone labels. They are not official challenge-toolkit scores. A fixed seed does not guarantee identical MPS runs because the installed backend warns about a nondeterministic operation.

## CPU timing

Run from the repository root while other compute work is idle:

```sh
uv run --locked python scripts/benchmark.py \
  --model results/phase1-ten-epochs/weights/best.pt \
  --image datasets/VisDrone/images/val/0000001_03999_d_0000007.jpg \
  --runs 100 --csv results/benchmarks.csv
```

One invocation appends one CSV row. Raw sorted samples, model and image hashes, predictor settings, and source metadata are saved under `results/benchmark-runs/`. Model size uses decimal MB; image size is the configured target, and rectangular padding may use a smaller tensor. The thread-count column records actual PyTorch intra-op threads after predictor setup.

Liam's timing loop measures the complete CPU prediction call, including preprocessing and postprocessing. Console logging is disabled; file loading and warm-up are outside the measurement. Its p95 convention is the sorted sample at index `int(0.95 * N)`. FPS is the reciprocal of mean call latency, not measured video-pipeline throughput. Repeating one image does not establish representative latency across the validation set. Superseded measurements remain in the CSV with their status and reason.

## Visual comparison

Use the same saved source frame for both models:

```sh
uv run --locked python scripts/compare.py \
  --before models/yolo26n.pt \
  --after results/phase1-ten-epochs/weights/best.pt \
  --image results/phase0-baseline/source-frame-000176.jpg \
  --output results/before-after
```

The output directory must be new. It receives a side-by-side PNG, native prediction JSON for each model, and shared settings and provenance. This is a qualitative illustration; COCO and VisDrone use different class vocabularies, and additional boxes do not by themselves establish higher accuracy.

## Data and licensing

The smoke-test clip is [Flying through downtown Woodstock](https://connecthv.com/drone-footage/flying-through-downtown-woodstock), created by Jordan Koschei and published by ConnectHV. Its [drone footage library](https://connecthv.com/drone-footage) releases the footage under CC0. The command uses the site's preview MP4. See [data/SOURCES.md](data/SOURCES.md) for provenance and checksum. This clip is separate from VisDrone and is not evaluation data.

Fine-tuning uses [VisDrone](https://docs.ultralytics.com/datasets/detect/visdrone); its download sources are recorded in `data/SOURCES.md`, with exact sizes and checksums in `results/phase1-setup/dataset-manifest.json`. Ultralytics provides [AGPL and Enterprise licensing options](https://www.ultralytics.com/license); a proprietary deployment needs a separate licensing assessment. No dataset or model weights are included in this repository.

## Manual work

The supplied handwritten timing loop is in `scripts/timing.py`, with its logic and variable names preserved. The bandwidth calculation in `scripts/bandwidth.py` and `DECISIONS.md` remain placeholders for Liam to write by hand. Measured values in this README must come from the results CSV.
