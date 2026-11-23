"""Unit tests for Macenko stain and optical density normalizer."""

import unittest
import numpy as np
from src.preprocessing.macenko_normalizer import MacenkoNormalizer


class TestMacenkoNormalizer(unittest.TestCase):
    def setUp(self):
        self.normalizer = MacenkoNormalizer(beta=0.1, alpha=1.0)

    def test_od_roundtrip(self):
        img = np.random.randint(50, 220, (30, 30, 3), dtype=np.uint8)
        od = self.normalizer.rgb_to_od(img)
        reconstructed = self.normalizer.od_to_rgb(od)
        # Difference between original and reconstructed should be minimal
        diff = np.max(np.abs(img.astype(float) - reconstructed.astype(float)))
        self.assertLess(diff, 2.0)

    def test_normalize_preserves_shape_and_dtype(self):
        img = np.random.randint(20, 240, (40, 40, 3), dtype=np.uint8)
        normalized = self.normalizer.normalize(img)
        self.assertEqual(normalized.shape, img.shape)
        self.assertEqual(normalized.dtype, np.uint8)

    def test_low_od_image_handled_gracefully(self):
        # Very bright/white image with almost no absorption
        img = np.full((32, 32, 3), 254, dtype=np.uint8)
        normalized = self.normalizer.normalize(img)
        self.assertEqual(normalized.shape, (32, 32, 3))


if __name__ == "__main__":
    unittest.main()
