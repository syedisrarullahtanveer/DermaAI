import unittest
import numpy as np
from src.evaluation.youdens_j import YoudensJOptimizer

class TestYoudensJ(unittest.TestCase):
    def test_optimal_cutoff(self):
        opt = YoudensJOptimizer()
        y_t = np.array([1, 1, 0, 0])
        y_s = np.array([0.8, 0.9, 0.2, 0.1])
        res = opt.find_best_cutoff(y_t, y_s)
        self.assertEqual(res["max_j"], 1.0) # perfect separation

if __name__ == '__main__':
    unittest.main()
