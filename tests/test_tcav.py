import unittest
import numpy as np
from src.explainability.tcav import TCAVExtractor

class TestTCAV(unittest.TestCase):
    def test_tcav_scoring(self):
        tcav = TCAVExtractor()
        c_act = np.random.randn(10, 8) + 2.0
        nc_act = np.random.randn(10, 8) - 2.0
        tcav.fit_cav(c_act, nc_act)
        grads = np.random.randn(15, 8)
        score = tcav.compute_tcav_score(grads)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
