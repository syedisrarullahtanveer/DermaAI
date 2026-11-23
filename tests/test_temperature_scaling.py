import unittest
import numpy as np
from src.evaluation.temperature_scaling import TemperatureScaler

class TestTemperatureScaling(unittest.TestCase):
    def test_fit_and_calibrate(self):
        scaler = TemperatureScaler()
        logits = np.array([[2.0, 0.5], [0.2, 1.8], [3.0, 0.1]])
        labels = np.array([0, 1, 0])
        t = scaler.fit(logits, labels)
        self.assertGreater(t, 0.0)
        probs = scaler.calibrate(logits)
        self.assertAlmostEqual(np.sum(probs[0]), 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
