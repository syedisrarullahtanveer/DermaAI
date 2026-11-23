import unittest
import numpy as np
from src.evaluation.bootstrap_ci import BootstrapCIEstimator

class TestBootstrapCI(unittest.TestCase):
    def test_bootstrap_accuracy(self):
        def acc(t, p): return float(np.mean(t == p))
        estimator = BootstrapCIEstimator(n_bootstraps=100, seed=42)
        y_t = np.array([1, 1, 0, 0, 1, 0, 1, 1])
        y_p = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        res = estimator.estimate(acc, y_t, y_p)
        self.assertIn("ci_lower", res)
        self.assertIn("ci_upper", res)
        self.assertLessEqual(res["ci_lower"], res["mean"])
        self.assertGreaterEqual(res["ci_upper"], res["mean"])

if __name__ == '__main__':
    unittest.main()
