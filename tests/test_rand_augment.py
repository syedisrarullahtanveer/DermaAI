import unittest
import numpy as np
from src.preprocessing.rand_augment import RandAugmentDermoscopy

class TestRandAugment(unittest.TestCase):
    def test_augmentation_output(self):
        aug = RandAugmentDermoscopy(n=2, m=3, seed=123)
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = aug.augment(img)
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
