# Aerial edge bench

Experiments on the tradeoffs between aerial object detection, model size, CPU inference, and sending detections instead of video.

## Current scope

The initial setup runs a pretrained YOLO26 nano detector on a local drone video and saves an annotated copy. This is a visual smoke test. Fine-tuning, accuracy evaluation, latency measurement, quantization, and bandwidth analysis have not run.

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

After downloading the weights, run the integration check with `uv run --locked python -m unittest discover -s tests -v`. It verifies that the requested CPU thread count survives library setup and that a short video is saved in the requested directory and decodes.

## Data and licensing

The smoke-test clip is [Flying through downtown Woodstock](https://connecthv.com/drone-footage/flying-through-downtown-woodstock), created by Jordan Koschei and published by ConnectHV. Its [drone footage library](https://connecthv.com/drone-footage) releases the footage under CC0. The command uses the site's preview MP4. See [data/SOURCES.md](data/SOURCES.md) for provenance and checksum. This clip is separate from VisDrone and is not evaluation data.

The planned fine-tuning dataset is [VisDrone](https://docs.ultralytics.com/datasets/detect/visdrone), whose download and terms must be recorded when that phase begins. Ultralytics provides [AGPL and Enterprise licensing options](https://www.ultralytics.com/license); a proprietary deployment needs a separate licensing assessment. No dataset or model weights are included in this repository.

## Manual work

The timing loop in `scripts/timing.py`, the bandwidth calculation in `scripts/bandwidth.py`, and `DECISIONS.md` remain placeholders for Liam to write by hand. There are no benchmark results yet. Any later measured values in this README must come from the results CSV.
