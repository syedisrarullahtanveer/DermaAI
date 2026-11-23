import unittest
import numpy as np
from src.explainability.score_cam import ScoreCAM

class TestScoreCAM(unittest.TestCase):
    def test_heatmap_generation(self):
        cam = ScoreCAM(None, "conv")
        activations = np.ones((16, 16, 4), dtype=np.float32)
        weights = np.array([1.0, 2.0, 0.5, 0.0], dtype=np.float32)
        heatmap = cam.generate_heatmap(activations, weights)
        self.assertEqual(heatmap.shape, (16, 16))
        self.assertAlmostEqual(np.max(heatmap), 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
