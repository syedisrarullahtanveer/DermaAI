import unittest
import numpy as np
from src.preprocessing.local_contrast_norm import LocalContrastNormalizer

class TestLocalContrastNorm(unittest.TestCase):
    def test_normalization_shape(self):
        normer = LocalContrastNormalizer()
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        out = normer.normalize(img, ksize=5)
        self.assertEqual(out.shape, img.shape)
        self.assertTrue(np.all(np.isfinite(out)))

if __name__ == '__main__':
    unittest.main()
