import unittest
import numpy as np
from src.preprocessing.jpeg_quality import JPEGArtifactDetector

class TestJPEGArtifact(unittest.TestCase):
    def test_blockiness_score(self):
        detector = JPEGArtifactDetector()
        img = np.zeros((32, 32), dtype=np.uint8)
        score = detector.score_blockiness(img)
        self.assertAlmostEqual(score, 0.0, places=4)

if __name__ == '__main__':
    unittest.main()
