import unittest
import cv2
import numpy as np
from src.preprocessing.diameter_estimator import LesionDiameterEstimator

class TestDiameterEstimator(unittest.TestCase):
    def test_circle_diameter(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 40, 1, -1)
        est = LesionDiameterEstimator(mm_per_pixel=0.1) # 80 px * 0.1 = 8.0 mm
        res = est.estimate_diameter(mask)
        self.assertAlmostEqual(res["diameter_pixels"], 80.0, places=0)
        self.assertTrue(res["exceeds_6mm"])

if __name__ == '__main__':
    unittest.main()
