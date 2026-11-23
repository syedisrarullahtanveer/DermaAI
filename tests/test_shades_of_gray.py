"""Unit tests for Shades of Gray color constancy."""

import unittest
import numpy as np
from src.preprocessing.shades_of_gray import ShadesOfGrayConstancy


class TestShadesOfGrayConstancy(unittest.TestCase):
    def setUp(self):
        self.transformer = ShadesOfGrayConstancy(power=6)

    def test_invalid_power_raises(self):
        with self.assertRaises(ValueError):
            ShadesOfGrayConstancy(power=0)

    def test_shape_and_dtype_preservation_uint8(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        transformed = self.transformer.transform(img)
        self.assertEqual(transformed.shape, (64, 64, 3))
        self.assertEqual(transformed.dtype, np.uint8)

    def test_shape_and_dtype_preservation_float(self):
        img = np.random.uniform(0.0, 1.0, (48, 48, 3)).astype(np.float32)
        transformed = self.transformer.transform(img)
        self.assertEqual(transformed.shape, (48, 48, 3))
        self.assertEqual(transformed.dtype, np.float32)

    def test_color_cast_balancing(self):
        # Create an image with an artificial reddish tint (higher red channel)
        img = np.zeros((50, 50, 3), dtype=np.float64)
        img[:, :, 0] = 0.9  # Red
        img[:, :, 1] = 0.3  # Green
        img[:, :, 2] = 0.3  # Blue

        transformed = self.transformer.transform(img)
        # In transformed image, channel averages should be closer to each other
        ch_means = np.mean(transformed, axis=(0, 1))
        # Distance between channel means should decrease
        orig_spread = np.std([0.9, 0.3, 0.3])
        new_spread = np.std(ch_means)
        self.assertLess(new_spread, orig_spread)

    def test_zero_image_does_not_crash(self):
        img = np.zeros((32, 32, 3), dtype=np.uint8)
        transformed = self.transformer.transform(img)
        self.assertEqual(transformed.shape, (32, 32, 3))
        self.assertEqual(int(transformed.sum()), 0)


if __name__ == "__main__":
    unittest.main()
