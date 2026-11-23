import unittest
from src.evaluation.diagnostic_ratios import DiagnosticRatioCalculator

class TestDiagnosticRatios(unittest.TestCase):
    def test_ratio_computations(self):
        calc = DiagnosticRatioCalculator()
        res = calc.compute(tp=90, fp=10, fn=10, tn=90)
        self.assertGreater(res["lr_positive"], 1.0)
        self.assertLess(res["lr_negative"], 1.0)
        self.assertGreater(res["diagnostic_odds_ratio"], 1.0)

if __name__ == '__main__':
    unittest.main()
