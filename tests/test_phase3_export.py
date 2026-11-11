import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import TensorProto, helper

from scripts.export_phase3 import inspect_graph


class ExportInspectionTest(unittest.TestCase):
    def test_shape_and_precision_cannot_be_mislabeled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            graph = helper.make_graph(
                [helper.make_node("Identity", ["images"], ["output"])], "test",
                [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 640, 640])],
            )
            onnx.save(helper.make_model(graph), path)
            result = inspect_graph(path, 640, 32)
            self.assertEqual(result["input_shape"], [1, 3, 640, 640])
            self.assertEqual(result["int8_initializers"], 0)
            with self.assertRaisesRegex(ValueError, "shape"):
                inspect_graph(path, 960, 32)
            with self.assertRaisesRegex(ValueError, "INT8"):
                inspect_graph(path, 640, 8)

    def test_quantized_graph_is_not_reported_as_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            graph = helper.make_graph([
                helper.make_node("QuantizeLinear", ["images", "scale", "zero"], ["quantized"]),
                helper.make_node("DequantizeLinear", ["quantized", "scale", "zero"], ["output"]),
            ], "test", [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 640, 640])],
                [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 640, 640])],
                [helper.make_tensor("scale", TensorProto.FLOAT, [], [0.1]),
                 helper.make_tensor("zero", TensorProto.INT8, [], [0])])
            onnx.save(helper.make_model(graph), path)
            self.assertGreater(inspect_graph(path, 640, 8)["node_counts"]["QuantizeLinear"], 0)
            with self.assertRaisesRegex(ValueError, "FP32"):
                inspect_graph(path, 640, 32)


if __name__ == "__main__":
    unittest.main()
