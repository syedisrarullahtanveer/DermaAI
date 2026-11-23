import unittest
import numpy as np
from src.evaluation.cost_matrix import CostSensitiveEvaluator

class TestCostSensitive(unittest.TestCase):
    def test_penalized_cost(self):
        # 2x2 matrix: TN=0, FP=1, FN=10 (missed cancer), TP=0
        cm = np.array([[0, 1], [10, 0]])
        evaluator = CostSensitiveEvaluator(cm)
        y_t = np.array([1, 0])
        y_p = np.array([0, 1]) # FN + FP
        self.assertEqual(evaluator.compute_total_cost(y_t, y_p), 11.0)

if __name__ == '__main__':
    unittest.main()
