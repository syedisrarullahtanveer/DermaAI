import unittest
from src.optimization.onnx_exporter import ONNXModelExporter

class TestONNXExporter(unittest.TestCase):
    def test_export_summary(self):
        exporter = ONNXModelExporter()
        s = exporter.export_summary("EdgeCNN", (224, 224, 3))
        self.assertEqual(s["opset_version"], 15)
        self.assertIn("CUDAExecutionProvider", s["supported_providers"])

if __name__ == '__main__':
    unittest.main()
