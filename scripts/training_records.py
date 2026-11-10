"""Epoch timing and sampled MPS memory, independent of the training device.

epoch-NNN.json and epoch-NNN-per-class.json are native validation metrics.
final-validation.json and final-per-class.json describe final best validation.
epoch-records.json contains measured epoch telemetry, not final validation time.
"""

import json
from pathlib import Path
import resource
import sys
import threading
import time
from typing import Callable


class EpochRecorder:
    def __init__(self, output_dir: Path, synchronize: Callable,
                 memory_usage: Callable, interval_seconds: float = 0.1,
                 clock: Callable = time.perf_counter):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.synchronize = synchronize
        self.memory_usage = memory_usage
        self.interval_seconds = interval_seconds
        self.clock = clock
        self.epochs = []
        self._active = None
        self._thread = None
        self._stop = threading.Event()
        self._error = None

    def _sample(self):
        values = self.memory_usage()
        for key in ("mps_tensor_bytes", "mps_driver_bytes"):
            value = values[key]
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"Invalid memory counter {key}: {value!r}")
            self._peaks[key] = max(self._peaks.get(key, value), value)
        self._samples += 1

    def _sample_loop(self):
        try:
            while not self._stop.wait(self.interval_seconds):
                self._sample()
        except Exception as error:
            self._error = error
            self._stop.set()

    def _stop_sampler(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError("MPS memory sampler failed") from error

    def on_epoch_start(self, trainer):
        if self._active is not None:
            raise RuntimeError("An epoch is already active")
        self.synchronize()
        self._start = self.clock()
        self._peaks = {}
        self._samples = 0
        self._sample()
        self._active = trainer.epoch + 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True,
                                        name="epoch-memory-sampler")
        self._thread.start()

    def _write_json(self, name, value):
        (self.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def _save_metrics(self, trainer, final=False):
        metrics = trainer.validator.metrics
        stem = "final-validation" if final else f"epoch-{self._active:03d}"
        per_class = "final-per-class.json" if final else f"{stem}-per-class.json"
        native_json = metrics.to_json(decimals=8)
        json.loads(native_json)  # Reject an invalid export before publishing files.
        self._write_json(f"{stem}.json", {key: float(value) for key, value in metrics.results_dict.items()})
        (self.output_dir / per_class).write_text(native_json + "\n")

    def on_fit_epoch_end(self, trainer):
        if self._active is None:
            self._save_metrics(trainer, final=True)
            return
        self.synchronize()
        elapsed = self.clock() - self._start
        self._stop_sampler()
        self._sample()
        self._save_metrics(trainer)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        record = {
            "epoch": self._active,
            "wall_seconds": elapsed,
            "timing_scope": "synchronized epoch start through training, validation and checkpoint saving; excludes final best validation",
            "sampled_peak_mps_tensor_bytes": self._peaks["mps_tensor_bytes"],
            "sampled_peak_mps_driver_bytes": self._peaks["mps_driver_bytes"],
            "memory_sample_interval_seconds": self.interval_seconds,
            "memory_sample_count": self._samples,
            "memory_limitations": "Sampled high-water marks are lower bounds and can miss transients. Driver bytes include cached and framework allocations; counters must not be added.",
            "process_lifetime_peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024),
            "rss_scope": "process lifetime high-water mark, not epoch-local or GPU-only",
        }
        self.epochs.append(record)
        self._active = None
        self._write_json("epoch-records.json", self.epochs)

    def close(self):
        """Stop sampling; surface background failures even on an aborted run."""
        try:
            self._stop_sampler()
        finally:
            self._active = None
