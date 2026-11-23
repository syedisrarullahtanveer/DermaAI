import unittest
import numpy as np
from src.evaluation.adaptive_ece import AdaptiveECE

class TestAdaptiveECE(unittest.TestCase):
    def test_perfect_calibration(self):
        evaluator = AdaptiveECE(num_bins=5)
        conf = np.array([0.9, 0.9, 0.1, 0.1])
        acc = np.array([0.9, 0.9, 0.1, 0.1])
        ece = evaluator.compute(conf, acc)
        self.assertAlmostEqual(ece, 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
