"""Regression checks for the adapter around the unchanged handwritten loop."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor, DetectionValidator

from scripts.phase3_runtime import ShapeAudit, make_guarded_types


def tensor(size=640, *, shape=None, dtype=torch.float32, device="cpu"):
    return SimpleNamespace(shape=shape or (1, 3, size, size), dtype=dtype,
                           device=torch.device(device))


class Phase3RuntimeTests(unittest.TestCase):
    def predictor_type(self, size=640):
        configured, _ = make_guarded_types(size)
        with patch.object(YOLO, "predict", return_value="result") as parent:
            configured.predict(object.__new__(configured), "image")
        return parent.call_args.kwargs["predictor"]

    def test_shape_audit_counts_only_valid_cpu_fp32_square_inputs(self):
        audit = ShapeAudit(640)
        audit.observe(tensor())
        audit.observe(tensor())
        expected = {"expected_shape": [1, 3, 640, 640],
                    "observed_shapes": [[1, 3, 640, 640]], "calls": 2}
        self.assertEqual(audit.as_dict(), expected)
        for bad in (tensor(shape=(1, 3, 384, 640)), tensor(shape=(2, 3, 640, 640)),
                    tensor(dtype=torch.float16), tensor(device="meta")):
            with self.subTest(tensor=bad):
                with self.assertRaises(ValueError):
                    audit.observe(bad)
                self.assertEqual(audit.as_dict(), expected)

    def test_predict_forces_matrix_settings_and_rejects_other_devices(self):
        configured, _ = make_guarded_types(960)
        model = object.__new__(configured)
        with patch.object(YOLO, "predict", return_value="result") as parent:
            result = configured.predict(model, "image", imgsz=640, rect=True,
                                        nms=False, max_det=300, verbose=True,
                                        conf=0.25, device="cpu")
            self.assertEqual(result, "result")
            settings = parent.call_args.kwargs
            self.assertEqual(parent.call_args.args, ("image",))
            self.assertEqual({k: v for k, v in settings.items() if k != "predictor"}, {
                "imgsz": 960, "rect": False, "nms": None, "max_det": 902,
                "verbose": False, "conf": 0.25, "device": "cpu",
            })
            self.assertTrue(issubclass(settings["predictor"], DetectionPredictor))
            parent.reset_mock()
            with self.assertRaisesRegex(ValueError, "cpu"):
                configured.predict(model, "image", device="mps")
            parent.assert_not_called()

    def test_predictor_rejects_bad_input_or_threads_before_backend_call(self):
        guarded = self.predictor_type()
        with patch.object(DetectionPredictor, "__init__", return_value=None):
            predictor = guarded()
        with (patch.object(DetectionPredictor, "inference", return_value="predictions") as parent,
              patch("torch.get_num_threads", return_value=8)):
            for bad in (tensor(shape=(1, 3, 384, 640)), tensor(dtype=torch.float16)):
                with self.assertRaises(ValueError):
                    predictor.inference(bad)
            parent.assert_not_called()
            self.assertEqual(predictor.inference(tensor()), "predictions")
            self.assertEqual(predictor.shape_audit.as_dict()["calls"], 1)
            self.assertEqual(predictor.torch_threads, 8)
            parent.assert_called_once()
            parent.reset_mock()
            with patch("torch.get_num_threads", return_value=2):
                with self.assertRaisesRegex(RuntimeError, "threads"):
                    predictor.inference(tensor())
            parent.assert_not_called()

    def test_predictor_restores_threads_after_library_setup(self):
        guarded = self.predictor_type()
        with patch.object(DetectionPredictor, "__init__", return_value=None):
            predictor = guarded()
        state = {"threads": 8}

        def setup(*args, **kwargs):
            state["threads"] = 2

        def set_threads(count):
            state["threads"] = count

        with (patch.object(DetectionPredictor, "setup_model", side_effect=setup) as parent,
              patch("torch.set_num_threads", side_effect=set_threads),
              patch("torch.get_num_threads", side_effect=lambda: state["threads"])):
            predictor.setup_model("weights", verbose=False)
        parent.assert_called_once_with("weights", verbose=False)
        self.assertEqual(predictor.torch_threads, 8)
        self.assertEqual(state["threads"], 8)

    def test_validator_checks_preprocessed_tensor_and_keeps_callbacks_private(self):
        _, guarded = make_guarded_types(960)
        original_callbacks = {"on_val_start": []}

        def initialize(validator, *args, **kwargs):
            validator.callbacks = original_callbacks

        with patch.object(DetectionValidator, "__init__", initialize):
            validator = guarded()
        self.assertIs(guarded.last_instance, validator)
        self.assertEqual(original_callbacks, {"on_val_start": []})
        with (patch("torch.set_num_threads") as set_threads,
              patch("torch.get_num_threads", return_value=8)):
            validator.run_callbacks("on_val_start")
        set_threads.assert_called_once_with(8)
        self.assertEqual(validator.torch_threads, 8)
        processed = {"img": tensor(960)}
        with (patch.object(DetectionValidator, "preprocess", return_value=processed) as parent,
              patch("torch.get_num_threads", return_value=8)):
            self.assertIs(validator.preprocess({"raw": "input"}), processed)
            parent.assert_called_once_with({"raw": "input"})
            self.assertEqual(validator.shape_audit.as_dict()["calls"], 1)
            parent.return_value = {"img": tensor()}
            with self.assertRaises(ValueError):
                validator.preprocess({"raw": "input"})
            self.assertEqual(validator.shape_audit.as_dict()["calls"], 1)


if __name__ == "__main__":
    unittest.main()
