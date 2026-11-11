"""Create auditable latency/size versus accuracy charts from the measured matrix."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

if __package__:
    from .phase3_records import complete_matrix
else:
    from phase3_records import complete_matrix


FORMATS = {
    "pytorch_fp32": {"label": "PyTorch FP32", "color": "#96a9bb"},
    "onnx_fp32": {"label": "ONNX FP32", "color": "#afc0b3"},
    "onnx_int8": {"label": "ONNX INT8", "color": "#c9aa76"},
}
MARKERS = {640: "o", 960: "D"}
OFFSETS = {
    "mean_ms": {
        ("pytorch_fp32", 640): (14, -8, "left"),
        ("onnx_fp32", 640): (-14, 18, "right"),
        ("onnx_int8", 640): (14, 0, "left"),
        ("pytorch_fp32", 960): (-14, 16, "right"),
        ("onnx_fp32", 960): (-14, 16, "right"),
        ("onnx_int8", 960): (14, -4, "left"),
    },
    "model_size_mb": {
        ("pytorch_fp32", 640): (-14, -18, "right"),
        ("onnx_fp32", 640): (-14, 16, "right"),
        ("onnx_int8", 640): (-14, 8, "right"),
        ("pytorch_fp32", 960): (14, 16, "left"),
        ("onnx_fp32", 960): (-14, 16, "right"),
        ("onnx_int8", 960): (14, 0, "left"),
    },
}


def load_chart_data(csv_path: Path) -> dict:
    """Reject incomparable inputs and preserve every plotted value's CSV source."""
    rows = complete_matrix(csv_path)
    hardware = {row["hardware"] for row in rows}
    threads = {int(row["thread_count"]) for row in rows}
    if len(hardware) != 1 or len(threads) != 1:
        raise ValueError("Charts require one hardware and thread configuration")
    points = []
    for row in rows:
        byte_count, megabytes = int(row["model_size_bytes"]), float(row["model_size_mb"])
        if not math.isclose(megabytes, byte_count / 1_000_000, rel_tol=1e-12):
            raise ValueError("Model file size must use decimal MB consistent with bytes")
        points.append({
            "format": row["format"],
            "image_size": int(row["image_size"]),
            "actual_input_height": int(row["actual_input_height"]),
            "actual_input_width": int(row["actual_input_width"]),
            "mean_ms": float(row["mean_ms"]),
            "model_size_mb": megabytes,
            "model_size_bytes": byte_count,
            "val_map50_95": float(row["val_map50_95"]),
            "csv_values": row.copy(),
        })
    order = {name: index for index, name in enumerate(FORMATS)}
    points.sort(key=lambda p: (p["image_size"], order[p["format"]]))
    return {"hardware": hardware.pop(), "thread_count": threads.pop(),
            "trained_image_size": int(rows[0]["trained_image_size"]), "points": points}


def render_chart(data: dict, x_field: str, output_dir: Path) -> dict:
    """Render one scatter plot; return its exact values and presentation settings."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    speed_chart = x_field == "mean_ms"
    stem = "speed-vs-accuracy" if speed_chart else "file-size-vs-accuracy"
    title = "Latency versus accuracy" if speed_chart else "File size versus accuracy"
    hardware = f"{data['hardware']} CPU"
    subtitle = (f"{hardware} · mean full prediction-call latency" if speed_chart
                else "Recorded model files · VisDrone validation")
    xlabel = (f"Mean CPU latency (ms, lower is better) — {hardware}" if speed_chart
              else "Model file size (decimal MB)")
    ylabel = "Validation mAP50–95 (higher is better)"
    unit = "ms" if speed_chart else "MB"
    x_step = 10 if speed_chart else 2
    x_max = math.ceil(max(p[x_field] for p in data["points"]) / x_step) * x_step
    if speed_chart:
        x_max += x_step / 2
    y_min = max(0, math.floor((min(p["val_map50_95"] for p in data["points"]) - 0.01) * 100) / 100)
    y_max = min(1, math.ceil((max(p["val_map50_95"] for p in data["points"]) + 0.015) * 100) / 100)
    x_ticks = list(range(0, int(x_max) + 1, x_step))
    first_cent = int(round(y_min * 100))
    y_ticks = [index / 100 for index in range(first_cent + first_cent % 2,
                                             int(round(y_max * 100)) + 1, 2)]
    note = ("Latency is the mean of repeated predictions on one image; no uncertainty estimate."
            if speed_chart else
            "PyTorch checkpoints and ONNX graphs use different serialization formats; MB is decimal.")
    size_note = (f"Trained at {data['trained_image_size']}; larger inputs change inference size only. "
                 "Source: measured benchmark CSV.")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": "#e4e4df", "axes.labelcolor": "#e4e4df",
        "xtick.color": "#a0a29b", "ytick.color": "#a0a29b",
        "svg.fonttype": "none", "svg.hashsalt": "aerial-edge-bench-charts",
    }):
        figure, axis = plt.subplots(figsize=(12.8, 7.8), dpi=180)
        figure.set_facecolor("#111111")
        figure.subplots_adjust(left=0.105, right=0.97, bottom=0.215, top=0.775)
        axis.set_facecolor("#151515")
        axis.set_axisbelow(True)
        axis.grid(color="#313131", linewidth=0.6)
        for spine in axis.spines.values():
            spine.set_color("#313131")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set(xlim=(0, x_max), ylim=(y_min, y_max), xticks=x_ticks, yticks=y_ticks)
        axis.set_xlabel(xlabel, labelpad=16, fontsize=12.5)
        axis.set_ylabel(ylabel, labelpad=15, fontsize=12.5)
        axis.set_yticklabels([f"{tick:.2f}" for tick in y_ticks])
        axis.tick_params(axis="both", length=0, pad=9)
        figure.text(0.105, 0.94, title, fontsize=21, weight="bold", va="top")
        figure.text(0.105, 0.89, subtitle, fontsize=13, color="#a0a29b", va="top")
        legend_handles = [Line2D([], [], color=item["color"], linewidth=3, label=item["label"])
                          for item in FORMATS.values()]
        legend_handles.extend(Line2D([], [], color="#e4e4df", marker=marker, linestyle="none",
                                     markersize=7, label=f"{size} × {size} input")
                              for size, marker in MARKERS.items())
        figure.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.098, 0.843),
                      ncol=5, frameon=False, fontsize=10.5, handlelength=1.3,
                      columnspacing=1.8, labelcolor="#e4e4df")
        annotations = []
        plotted = []
        for point in data["points"]:
            style = FORMATS[point["format"]]
            x, y = point[x_field], point["val_map50_95"]
            marker = MARKERS[point["image_size"]]
            axis.scatter(x, y, marker=marker, s=105, color=style["color"],
                         edgecolors="#e4e4df", linewidths=0.65, zorder=4)
            dx, dy, alignment = OFFSETS[x_field][(point["format"], point["image_size"])]
            label = f"{style['label']} · {point['image_size']}\n{x:.2f} {unit}"
            axis.annotate(label, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                          ha=alignment, va="bottom" if dy >= 0 else "top", fontsize=12,
                          color=style["color"], linespacing=1.35,
                          arrowprops={"arrowstyle": "-", "color": style["color"],
                                      "linewidth": 0.65, "alpha": 0.7},
                          bbox={"facecolor": "#151515", "edgecolor": "none", "pad": 1.8},
                          zorder=5)
            plotted.append({"format": point["format"], "image_size": point["image_size"],
                            "x": x, "y": y, "marker": marker, "color": style["color"]})
            annotations.append({"format": point["format"], "image_size": point["image_size"],
                                "text": label, "offset_points": [dx, dy],
                                "horizontal_alignment": alignment})
        figure.text(0.105, 0.093, note, fontsize=11, color="#a0a29b", va="top")
        figure.text(0.105, 0.055, size_note, fontsize=11, color="#a0a29b", va="top")
        files = []
        for extension in ("png", "svg"):
            path = output_dir / f"{stem}.{extension}"
            metadata = ({"Software": None} if extension == "png" else
                        {"Date": None, "Creator": None, "Title": title,
                         "Description": f"{subtitle}. {note} {size_note}"})
            figure.savefig(path, dpi=180, facecolor=figure.get_facecolor(), metadata=metadata)
            files.append({"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size})
        plt.close(figure)
    return {
        "title": title, "subtitle": subtitle, "x_field": x_field, "x_unit": unit,
        "y_field": "val_map50_95", "y_unit": "score on zero-to-one scale",
        "xlabel": xlabel, "ylabel": ylabel,
        "x_limits": [0, x_max], "y_limits": [y_min, y_max],
        "x_ticks": x_ticks, "y_ticks": y_ticks,
        "y_tick_labels": [f"{tick:.2f}" for tick in y_ticks],
        "figure_inches": [12.8, 7.8], "png_dpi": 180,
        "points": plotted, "annotations": annotations, "notes": [note, size_note],
        "error_bars": None, "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("results/phase3/benchmarks.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--manifest", type=Path, default=Path("results/phase5/chart-manifest.json"))
    args = parser.parse_args()
    data = load_chart_data(args.csv)
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    charts = [render_chart(data, field, args.output_dir) for field in ("mean_ms", "model_size_mb")]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_csv": args.csv.as_posix(),
        "source_csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "matplotlib_version": matplotlib.__version__,
        "decimal_mb_bytes": 1_000_000,
        "hardware": data["hardware"], "thread_count": data["thread_count"],
        "trained_image_size": data["trained_image_size"],
        "source_points": data["points"], "charts": charts,
        "limitations": ["Single-image mean latency is not video throughput or target-device performance.",
                        "No error bars or statistical equivalence claims are inferred.",
                        "PyTorch checkpoint serialization differs from ONNX graph serialization."],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Saved both charts and {args.manifest}")


if __name__ == "__main__":
    main()
