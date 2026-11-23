import unittest
import numpy as np
from src.preprocessing.cutout_augmentation import random_cutout, CutoutAugmentation

class TestCutoutAugmentation(unittest.TestCase):
    def test_single_cutout(self):
        img = np.ones((64, 64, 3), dtype=np.float32)
        out = random_cutout(img, num_patches=2, patch_size=16, fill_value=0.0)
        self.assertEqual(out.shape, (64, 64, 3))
        self.assertTrue(np.any(out == 0.0))

    def test_batch_cutout(self):
        batch = np.ones((4, 32, 32, 3), dtype=np.float32)
        augmentor = CutoutAugmentation(num_patches=1, patch_size=8, fill_value=0.0, seed=42)
        out = augmentor(batch)
        self.assertEqual(out.shape, (4, 32, 32, 3))

if __name__ == '__main__':
    unittest.main()
