import unittest
import numpy as np
from src.preprocessing.image_quality import is_blurry

class TestImageQuality(unittest.TestCase):
    def test_blurry(self):
        # Uniform gray image should be blurry (low variance)
        img = np.full((100,100,3), 128, dtype=np.uint8)
        self.assertTrue(is_blurry(img))
    def test_sharp(self):
        # Random noise image should not be blurry (high variance)
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(100,100,3), dtype=np.uint8)
        self.assertFalse(is_blurry(img))

if __name__ == '__main__':
    unittest.main()
