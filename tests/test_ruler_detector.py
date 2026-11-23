import unittest
import numpy as np
from src.preprocessing.ruler_detector import RulerArtifactDetector

class TestRulerDetector(unittest.TestCase):
    def test_detection_mask_shape(self):
        detector = RulerArtifactDetector()
        img = np.zeros((64, 64), dtype=np.uint8)
        mask = detector.detect_ruler_marks(img)
        self.assertEqual(mask.shape, img.shape)

if __name__ == '__main__':
    unittest.main()
