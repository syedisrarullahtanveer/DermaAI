import unittest
import numpy as np
from src.preprocessing.gel_bubble_filter import GelBubbleFilter

class TestGelBubble(unittest.TestCase):
    def test_filter_shape_preservation(self):
        flt = GelBubbleFilter()
        img = np.ones((64, 64, 3), dtype=np.uint8) * 150
        out = flt.filter_bubbles(img)
        self.assertEqual(out.shape, img.shape)

if __name__ == '__main__':
    unittest.main()
