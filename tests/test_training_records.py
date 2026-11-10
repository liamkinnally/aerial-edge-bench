"""CPU-only regression tests for measured epoch records."""

import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest

from scripts.training_records import EpochRecorder


class Metrics:
    results_dict = {"metrics/mAP50(B)": 0.25, "metrics/mAP50-95(B)": 0.1}

    def to_json(self, decimals=5):
        return '[{"Class":"pedestrian","mAP50":0.25,"mAP50-95":0.1}]'


class EpochRecorderTest(unittest.TestCase):
    def test_final_validation_does_not_replace_epoch_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            ticks = iter([10.0, 25.0])
            recorder = EpochRecorder(Path(directory), lambda: None,
                                     lambda: {"mps_tensor_bytes": 30, "mps_driver_bytes": 80},
                                     clock=lambda: next(ticks))
            trainer = SimpleNamespace(epoch=0, validator=SimpleNamespace(training=True, metrics=Metrics()))
            try:
                recorder.on_epoch_start(trainer)
                recorder.on_fit_epoch_end(trainer)
                trainer.epoch = 1
                trainer.validator.training = False
                trainer.validator.metrics = Metrics()
                trainer.validator.metrics.results_dict = {"metrics/mAP50(B)": 0.3}
                recorder.on_fit_epoch_end(trainer)
            finally:
                recorder.close()
            self.assertEqual(len(recorder.epochs), 1)
            self.assertEqual(recorder.epochs[0]["wall_seconds"], 15.0)
            self.assertEqual(json.loads((Path(directory) / "epoch-001.json").read_text())["metrics/mAP50(B)"], 0.25)
            self.assertEqual(json.loads((Path(directory) / "final-validation.json").read_text())["metrics/mAP50(B)"], 0.3)
            self.assertEqual(json.loads((Path(directory) / "epoch-001-per-class.json").read_text())[0]["Class"], "pedestrian")
            self.assertTrue((Path(directory) / "final-per-class.json").is_file())
            self.assertEqual(json.loads((Path(directory) / "epoch-records.json").read_text()), recorder.epochs)

    def test_background_sampling_preserves_distinct_high_water_marks(self):
        sampled = threading.Event()
        calls = 0

        def memory():
            nonlocal calls
            calls += 1
            if calls == 2:
                sampled.set()
                return {"mps_tensor_bytes": 70, "mps_driver_bytes": 100}
            return {"mps_tensor_bytes": 20, "mps_driver_bytes": 140}

        with tempfile.TemporaryDirectory() as directory:
            recorder = EpochRecorder(Path(directory), lambda: None, memory, interval_seconds=0.005)
            trainer = SimpleNamespace(epoch=0, validator=SimpleNamespace(training=True, metrics=Metrics()))
            try:
                recorder.on_epoch_start(trainer)
                self.assertTrue(sampled.wait(2), "background sampler did not run")
                recorder.on_fit_epoch_end(trainer)
            finally:
                recorder.close()
            record = recorder.epochs[0]
            self.assertEqual(record["sampled_peak_mps_tensor_bytes"], 70)
            self.assertEqual(record["sampled_peak_mps_driver_bytes"], 140)
            self.assertGreaterEqual(record["memory_sample_count"], 3)
            self.assertGreater(record["process_lifetime_peak_rss_bytes"], 0)

    def test_background_failure_is_surfaced_instead_of_reporting_zero(self):
        failed = threading.Event()
        calls = 0

        def memory():
            nonlocal calls
            calls += 1
            if calls > 1:
                failed.set()
                raise OSError("memory unavailable")
            return {"mps_tensor_bytes": 20, "mps_driver_bytes": 40}

        with tempfile.TemporaryDirectory() as directory:
            recorder = EpochRecorder(Path(directory), lambda: None, memory, interval_seconds=0.005)
            trainer = SimpleNamespace(epoch=0)
            recorder.on_epoch_start(trainer)
            self.assertTrue(failed.wait(2))
            with self.assertRaisesRegex(RuntimeError, "memory sampler failed"):
                recorder.close()
            self.assertEqual(recorder.epochs, [])


if __name__ == "__main__":
    unittest.main()
