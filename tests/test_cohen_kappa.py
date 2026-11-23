import unittest
import numpy as np
from src.evaluation.cohen_kappa import CohensKappaEvaluator

class TestCohensKappa(unittest.TestCase):
    def test_perfect_agreement(self):
        evaluator = CohensKappaEvaluator()
        y = np.array([0, 1, 2, 0, 1])
        score = evaluator.evaluate(y, y)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_one_hot_inputs(self):
        evaluator = CohensKappaEvaluator()
        y1 = np.array([[1, 0], [0, 1]])
        y2 = np.array([[1, 0], [0, 1]])
        self.assertAlmostEqual(evaluator.evaluate(y1, y2), 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
