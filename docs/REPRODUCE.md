# How to reproduce

[Back to the project overview](../README.md)

Run commands from the repository root. The training wrapper requires Apple Silicon with MPS and stops when it is unavailable. CPU export, validation and timing are separate commands. The pinned environment is in `uv.lock`; the recorded versions are in [the result summary](measurements/summary.json).

Set up a fresh checkout and download the demo clip:

```sh
export UV_CACHE_DIR="$PWD/.cache/uv"
uv sync --locked
mkdir -p data/raw
curl -fL \
  'https://media.connecthv.com/preview/Flying%20through%20downtown%20Woodstock.mp4' \
  -o data/raw/woodstock-drone-preview.mp4
uv run --locked python scripts/predict.py \
  --source data/raw/woodstock-drone-preview.mp4
```

The smoke-test command downloads the stock weights into `models/`. On a fresh checkout, train the checkpoint used by the matrix. If that checkpoint already exists, keep it and skip this command:

```sh
uv run --locked python scripts/train.py \
  --epochs 50 --batch 16 \
  --output results/phase1-fifty-epochs
```

Training downloads and converts VisDrone when needed, starts from pretrained weights, and saves native metrics and checkpoints. It uses the recorded image size and seed. The existing matrix runner expects the canonical checkpoint path shown below. Export, validation-row and comparison directories must be new; use a different output root for each repeat and a fresh CSV.

```sh
set -e
report_run_dir="results/reproduction"
report_weights="results/phase1-fifty-epochs/weights/best.pt"
report_frame="datasets/VisDrone/images/val/0000001_03999_d_0000007.jpg"

# Export FP32 first, then calibrated INT8 at each static shape.
for report_size in 640 960; do
  uv run --locked python scripts/export_phase3.py \
    --weights "$report_weights" --size "$report_size" --quantize 32 \
    --output "$report_run_dir/exports/fp32-$report_size"
done
for report_size in 640 960; do
  uv run --locked python scripts/export_phase3.py \
    --weights "$report_weights" --size "$report_size" --quantize 8 \
    --output "$report_run_dir/exports/int8-$report_size"
done

# Each row validates the full split, then runs the supplied timing loop.
for report_size in 640 960; do
  for report_format in pytorch_fp32 onnx_fp32 onnx_int8; do
    case "$report_format" in
      pytorch_fp32) report_model="$report_weights" ;;
      onnx_fp32) report_model="$report_run_dir/exports/fp32-$report_size/best.onnx" ;;
      onnx_int8) report_model="$report_run_dir/exports/int8-$report_size/best_int8.onnx" ;;
    esac
    uv run --locked python scripts/benchmark_phase3.py \
      --model "$report_model" --format "$report_format" --size "$report_size" \
      --image "$report_frame" --output "$report_run_dir/rows/$report_format-$report_size" \
      --csv "$report_run_dir/benchmarks.csv"
  done
done
```

Calibration uses the complete VisDrone training split, with a saved image inventory separate from validation. The wrappers use the pinned `quantize` API. Run the matrix sequentially while other compute work is idle. A quiet-machine or shape-check failure stops the command; preserve the failed output before retrying in a new directory. If an INT8 export or validation still fails after two distinct attempts, retain the errors and stop rather than substitute another precision.

Save the qualitative comparison and both bandwidth baselines in the same new output root:

```sh
uv run --locked python scripts/compare.py \
  --before models/yolo26n.pt --after "$report_weights" \
  --image docs/assets/source-frame.jpg --output "$report_run_dir/before-after"
uv run --locked python scripts/bandwidth.py \
  --model "$report_weights" --clip data/raw/woodstock-drone-preview.mp4 \
  --output "$report_run_dir/bandwidth.json"
```

The bandwidth command uses the clip's recorded frame rate and refuses to overwrite an existing result. The supplied source frame is retained with its original hash in the report summary.

Regenerate the charts and run the checks after the pretrained weights are present:

```sh
uv run --locked python scripts/plot_results.py \
  --csv docs/measurements/benchmarks.csv --output-dir docs/assets
uv run --locked python -m unittest discover -s tests -v
```

For charts of a new run, use that run's CSV and a separate asset directory. The versionable report contains selected results and images; model weights, datasets and full run logs remain local and ignored. Full original records live under `results/phase3/` and `results/phase4/`. The original dataset download manifest under `results/phase1-setup/` is retained evidence, not an output automatically recreated by these commands.
