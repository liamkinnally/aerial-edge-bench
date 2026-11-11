"""Configure and audit matrix runs outside the unchanged handwritten timing loop.

The caller binds ConfiguredYOLO only while running the loop in an isolated
process. The same configuration and tensor-guard overhead applies to each row.
"""


class ShapeAudit:
    """Count valid CPU FP32 inputs and reject a mismatch before inference."""

    def __init__(self, image_size: int):
        self.expected_shape = (1, 3, image_size, image_size)
        self.shapes = set()
        self.count = 0

    def observe(self, tensor):
        import torch

        shape = tuple(tensor.shape)
        if shape != self.expected_shape:
            raise ValueError(f"Input shape {shape} differs from required {self.expected_shape}")
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise ValueError(f"Input must be CPU float32, got {tensor.device} {tensor.dtype}")
        self.shapes.add(shape)
        self.count += 1

    def as_dict(self):
        return {
            "expected_shape": list(self.expected_shape),
            "observed_shapes": [list(shape) for shape in sorted(self.shapes)],
            "calls": self.count,
        }


def make_guarded_types(image_size: int, threads: int = 8, max_det: int = 902):
    """Return configured YOLO and validator classes for one fixed-size row."""
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionPredictor, DetectionValidator

    def check_threads(owner):
        actual = torch.get_num_threads()
        if actual != threads:
            raise RuntimeError(f"PyTorch threads changed: expected {threads}, observed {actual}")
        owner.torch_threads = actual

    class GuardedPredictor(DetectionPredictor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape_audit = ShapeAudit(image_size)
            self.torch_threads = None

        def setup_model(self, model, verbose=True):
            super().setup_model(model, verbose=verbose)
            # Ultralytics lowers CPU preprocessing threads for ONNX at setup.
            # Restore the common count during the first untimed warm-up call.
            torch.set_num_threads(threads)
            check_threads(self)

        def inference(self, im, *args, **kwargs):
            check_threads(self)
            self.shape_audit.observe(im)
            return super().inference(im, *args, **kwargs)

    class ConfiguredYOLO(YOLO):
        def predict(self, *args, **kwargs):
            if "device" in kwargs and str(kwargs["device"]) != "cpu":
                raise ValueError("Phase 3 prediction requires device='cpu'")
            # Explicit kwargs win over Model.predict's built-in rect=True.
            kwargs.update(imgsz=image_size, rect=False, nms=None, max_det=max_det,
                          verbose=False, device="cpu", predictor=GuardedPredictor)
            return super().predict(*args, **kwargs)

    class GuardedValidator(DetectionValidator):
        last_instance = None

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.shape_audit = ShapeAudit(image_size)
            self.torch_threads = None
            self.callbacks = {event: list(items) for event, items in self.callbacks.items()}
            self.add_callback("on_val_start", self._configure_threads)
            type(self).last_instance = self

        def _configure_threads(self, validator):
            torch.set_num_threads(threads)
            check_threads(self)

        def preprocess(self, batch):
            batch = super().preprocess(batch)
            check_threads(self)
            self.shape_audit.observe(batch["img"])
            return batch

    return ConfiguredYOLO, GuardedValidator
