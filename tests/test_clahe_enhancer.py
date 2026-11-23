"""Unit tests for CLAHE lesion contrast enhancement."""

import unittest
import numpy as np
from src.preprocessing.clahe_enhancer import CLAHEEnhancer


class TestCLAHEEnhancer(unittest.TestCase):
    def setUp(self):
        self.enhancer = CLAHEEnhancer(clip_limit=3.0, tile_grid_size=(4, 4))

    def test_invalid_parameters_raise(self):
        with self.assertRaises(ValueError):
            CLAHEEnhancer(clip_limit=-1.0)
        with self.assertRaises(ValueError):
            CLAHEEnhancer(tile_grid_size=(0, 4))

    def test_rgb_uint8_enhancement(self):
        img = np.random.randint(40, 180, (64, 64, 3), dtype=np.uint8)
        enhanced = self.enhancer.enhance(img)
        self.assertEqual(enhanced.shape, (64, 64, 3))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_float_image_enhancement(self):
        img = np.random.uniform(0.2, 0.8, (32, 32, 3)).astype(np.float32)
        enhanced = self.enhancer.enhance(img)
        self.assertEqual(enhanced.shape, (32, 32, 3))
        self.assertEqual(enhanced.dtype, np.float32)
        self.assertTrue(np.all(enhanced >= 0.0) and np.all(enhanced <= 1.0))

    def test_grayscale_image_enhancement(self):
        img = np.random.randint(20, 200, (48, 48), dtype=np.uint8)
        enhanced = self.enhancer.enhance(img)
        self.assertEqual(enhanced.shape, (48, 48))
        self.assertEqual(enhanced.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
