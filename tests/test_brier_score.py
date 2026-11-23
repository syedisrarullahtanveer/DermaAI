import unittest
import numpy as np
from src.evaluation.brier_score import MultiClassBrierScore

class TestBrierScore(unittest.TestCase):
    def test_perfect_score(self):
        brier = MultiClassBrierScore()
        y_t = np.array([0, 1])
        y_p = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(brier.compute(y_t, y_p), 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
