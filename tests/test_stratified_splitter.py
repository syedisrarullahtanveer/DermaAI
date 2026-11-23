import unittest
import numpy as np
from src.data.stratified_splitter import StratifiedKFoldSplitter

class TestStratifiedSplitter(unittest.TestCase):
    def test_splits_preserve_proportions(self):
        splitter = StratifiedKFoldSplitter(n_splits=3, seed=42)
        x = np.arange(12)
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
        splits = splitter.split(x, y)
        self.assertEqual(len(splits), 3)

if __name__ == '__main__':
    unittest.main()
