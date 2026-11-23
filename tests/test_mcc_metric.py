import unittest
import numpy as np
from src.evaluation.mcc_metric import MulticlassMCC

class TestMCC(unittest.TestCase):
    def test_perfect_prediction(self):
        mcc = MulticlassMCC()
        y = np.array([0, 1, 1, 2, 0])
        score = mcc.compute(y, y)
        self.assertAlmostEqual(score, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
