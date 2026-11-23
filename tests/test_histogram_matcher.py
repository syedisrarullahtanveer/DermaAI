import unittest
import numpy as np
from src.preprocessing.histogram_matcher import HistogramMatcher

class TestHistogramMatcher(unittest.TestCase):
    def test_match_shape_and_dtype(self):
        matcher = HistogramMatcher()
        src = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        tmpl = np.random.randint(50, 150, (16, 16, 3), dtype=np.uint8)
        out = matcher.match(src, tmpl)
        self.assertEqual(out.shape, src.shape)
        self.assertEqual(out.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()
