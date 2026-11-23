import unittest
import numpy as np
from src.evaluation.net_benefit import NetBenefitAnalyzer

class TestNetBenefit(unittest.TestCase):
    def test_net_benefit_calculation(self):
        analyzer = NetBenefitAnalyzer()
        y_t = np.array([1, 1, 0, 0])
        y_p = np.array([0.9, 0.8, 0.1, 0.2])
        nb = analyzer.compute_net_benefit(y_t, y_p, threshold=0.5)
        # tp=2/4 = 0.5, fp=0, weight=1.0 -> nb = 0.5
        self.assertAlmostEqual(nb, 0.5, places=4)

if __name__ == '__main__':
    unittest.main()
