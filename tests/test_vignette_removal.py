"""Unit tests for VignetteCornerRemover."""

import unittest
import numpy as np
from src.preprocessing.vignette_removal import VignetteCornerRemover


class TestVignetteCornerRemover(unittest.TestCase):
    def setUp(self):
        self.remover = VignetteCornerRemover(intensity_threshold=20, min_radius_ratio=0.3)

    def test_vignette_cropped_correctly(self):
        # Create an image with black margins and a bright center region
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[20:80, 20:80] = 180  # Central illuminated field

        cropped = self.remover.crop_vignette(img)
        # Cropped image should have smaller dimensions and retain the bright region
        self.assertLessEqual(cropped.shape[0], 100)
        self.assertLessEqual(cropped.shape[1], 100)
        self.assertGreaterEqual(cropped.shape[0], 50)
        self.assertGreaterEqual(cropped.shape[1], 50)
        self.assertEqual(cropped.shape[2], 3)

    def test_uniform_bright_image_retained(self):
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        cropped = self.remover.crop_vignette(img)
        self.assertEqual(cropped.shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
