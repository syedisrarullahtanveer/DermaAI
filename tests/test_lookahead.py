import unittest
import numpy as np
from src.training.lookahead import LookaheadOptimizer

class TestLookahead(unittest.TestCase):
    def test_update_at_k(self):
        opt = LookaheadOptimizer(alpha=0.5, k=2)
        initial = [np.array([0.0, 0.0])]
        opt.initialize(initial)
        
        # Step 1: not synchronized
        w1 = opt.step([np.array([2.0, 2.0])])
        self.assertEqual(w1[0][0], 2.0)
        
        # Step 2: synchronized: 0 + 0.5*(2 - 0) = 1.0
        w2 = opt.step([np.array([2.0, 2.0])])
        self.assertEqual(w2[0][0], 1.0)

if __name__ == '__main__':
    unittest.main()
