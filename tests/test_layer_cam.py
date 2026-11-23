import unittest
import numpy as np
from src.explainability.layer_cam import LayerCAM

class TestLayerCAM(unittest.TestCase):
    def test_layer_cam_computation(self):
        cam_gen = LayerCAM()
        act = np.ones((16, 16, 8), dtype=np.float32)
        grad = np.random.randn(16, 16, 8).astype(np.float32)
        cam = cam_gen.compute_cam(act, grad)
        self.assertEqual(cam.shape, (16, 16))
        self.assertLessEqual(np.max(cam), 1.0)

if __name__ == '__main__':
    unittest.main()
