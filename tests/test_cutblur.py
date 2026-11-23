import unittest
import numpy as np
from src.preprocessing.cutblur import CutBlurAugmentation

class TestCutBlur(unittest.TestCase):
    def test_cutblur_application(self):
        cb = CutBlurAugmentation(blur_kernel=3, downscale_factor=2)
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        out = cb.apply(img, bbox=(10, 10, 20, 20))
        self.assertEqual(out.shape, img.shape)

if __name__ == '__main__':
    unittest.main()
