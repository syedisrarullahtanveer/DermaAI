import unittest
import cv2
import numpy as np
from src.preprocessing.border_irregularity import BorderIrregularityExtractor

class TestBorderIrregularity(unittest.TestCase):
    def test_circle_compactness_close_to_one(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 1, -1)
        extractor = BorderIrregularityExtractor()
        score = extractor.compute_compactness(mask)
        self.assertAlmostEqual(score, 1.0, places=1)

if __name__ == '__main__':
    unittest.main()
