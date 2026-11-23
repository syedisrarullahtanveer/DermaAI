import unittest
import numpy as np
from src.evaluation.partial_auc import PartialAUCEvaluator

class TestPartialAUC(unittest.TestCase):
    def test_pauc_range(self):
        evaluator = PartialAUCEvaluator(min_specificity=0.8)
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_score = np.array([0.9, 0.8, 0.2, 0.1, 0.7, 0.3])
        pauc = evaluator.compute_pauc(y_true, y_score)
        self.assertGreaterEqual(pauc, 0.0)
        self.assertLessEqual(pauc, 1.0)

if __name__ == '__main__':
    unittest.main()
