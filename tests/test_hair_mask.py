"""Unit tests for HairMaskGenerator."""

import unittest
import numpy as np
from src.preprocessing.hair_mask import HairMaskGenerator


class TestHairMaskGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = HairMaskGenerator(kernel_size=5, threshold=15)

    def test_invalid_kernel_size_raises(self):
        with self.assertRaises(ValueError):
            HairMaskGenerator(kernel_size=4)

    def test_detects_dark_line_hair(self):
        # Create light skin background
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        # Draw a thin dark line simulating a hair fiber
        img[30:32, :, :] = 20

        mask = self.generator.generate_mask(img)
        self.assertEqual(mask.shape, (64, 64))
        self.assertEqual(mask.dtype, np.uint8)
        # Line area should be detected as 255
        self.assertGreater(np.sum(mask[30:32, :] == 255), 50)
        # Peripheral area should be 0
        self.assertEqual(np.sum(mask[:20, :] == 255), 0)

    def test_uniform_image_has_empty_mask(self):
        img = np.full((32, 32, 3), 150, dtype=np.uint8)
        mask = self.generator.generate_mask(img)
        self.assertEqual(int(np.sum(mask)), 0)


if __name__ == "__main__":
    unittest.main()
