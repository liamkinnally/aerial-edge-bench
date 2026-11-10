# Aerial edge bench

Experiments on the tradeoffs between aerial object detection, model size, CPU inference, and sending detections instead of video.

## Current scope

The initial setup runs a pretrained YOLO26 nano detector on a local drone video and saves an annotated copy. The MPS fine-tuning pilot and short VisDrone training run are also complete, with weights and native validation outputs retained locally under `results/phase1-pilot/` and `results/phase1-ten-epochs/`. Latency measurement, quantization, export, and bandwidth analysis have not run.

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

The training directory contains `weights/best.pt`, `weights/last.pt`, the native `results.csv`, per-epoch validation JSON, and separate final-best aggregate and per-class validation JSON. `run-metadata.json` records the seed, device, package versions, Git revision, and source hashes. `epoch-records.json` includes validation and checkpoint saving in each epoch's wall time; it labels sampled MPS memory separately from process peak resident memory.

These are native Ultralytics validation results on converted VisDrone labels. They are not official challenge-toolkit scores. A fixed seed does not guarantee identical MPS runs because the installed backend warns about a nondeterministic operation.

## Data and licensing

The smoke-test clip is [Flying through downtown Woodstock](https://connecthv.com/drone-footage/flying-through-downtown-woodstock), created by Jordan Koschei and published by ConnectHV. Its [drone footage library](https://connecthv.com/drone-footage) releases the footage under CC0. The command uses the site's preview MP4. See [data/SOURCES.md](data/SOURCES.md) for provenance and checksum. This clip is separate from VisDrone and is not evaluation data.

Fine-tuning uses [VisDrone](https://docs.ultralytics.com/datasets/detect/visdrone); its download sources are recorded in `data/SOURCES.md`, with exact sizes and checksums in `results/phase1-setup/dataset-manifest.json`. Ultralytics provides [AGPL and Enterprise licensing options](https://www.ultralytics.com/license); a proprietary deployment needs a separate licensing assessment. No dataset or model weights are included in this repository.

## Manual work

The timing loop in `scripts/timing.py`, the bandwidth calculation in `scripts/bandwidth.py`, and `DECISIONS.md` remain placeholders for Liam to write by hand. There are no benchmark results yet. Any later measured values in this README must come from the results CSV.
