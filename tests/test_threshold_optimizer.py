import unittest
import numpy as np
from src.evaluation.threshold_optimizer import ClinicalThresholdOptimizer

class TestThresholdOptimizer(unittest.TestCase):
    def test_optimization_targets_sensitivity(self):
        opt = ClinicalThresholdOptimizer(target_sensitivity=0.90)
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        y_prob = np.array([0.9, 0.85, 0.8, 0.75, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
        res = opt.optimize(y_true, y_prob)
        self.assertGreaterEqual(res["sensitivity"], 0.90)
        self.assertGreater(res["threshold"], 0.0)

if __name__ == '__main__':
    unittest.main()
