import unittest
import numpy as np
from src.uncertainty.conformal import ConformalPredictionEvaluator

class TestConformal(unittest.TestCase):
    def test_prediction_sets(self):
        conformal = ConformalPredictionEvaluator(alpha=0.1)
        cal_p = np.array([[0.9, 0.1], [0.8, 0.2], [0.95, 0.05]])
        cal_y = np.array([0, 0, 0])
        conformal.calibrate(cal_p, cal_y)
        
        test_p = np.array([[0.85, 0.15], [0.5, 0.5]])
        sets = conformal.predict_sets(test_p)
        self.assertIn(0, sets[0])

if __name__ == '__main__':
    unittest.main()
