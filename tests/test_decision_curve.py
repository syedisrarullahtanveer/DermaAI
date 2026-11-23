"""Unit tests for DecisionCurveAnalyzer."""

import unittest
import numpy as np
from src.evaluation.decision_curve import DecisionCurveAnalyzer


class TestDecisionCurveAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = DecisionCurveAnalyzer(thresholds=[0.1, 0.2, 0.5])

    def test_perfect_model_net_benefit_equals_prevalence(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.95, 0.05, 0.02])

        res = self.analyzer.compute_net_benefit(y_true, y_prob)
        self.assertEqual(res["total_samples"], 4)
        self.assertAlmostEqual(res["prevalence"], 0.5)

        # For perfect model with threshold 0.2: TP=2, FP=0 -> Net Benefit = 2/4 - 0 = 0.5
        point_02 = [p for p in res["curve_points"] if abs(p["threshold"] - 0.2) < 1e-4][0]
        self.assertAlmostEqual(point_02["net_benefit_model"], 0.5)

    def test_treat_none_benefit_is_zero(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.6, 0.4, 0.8, 0.2])
        res = self.analyzer.compute_net_benefit(y_true, y_prob)
        for pt in res["curve_points"]:
            self.assertEqual(pt["net_benefit_none"], 0.0)


if __name__ == "__main__":
    unittest.main()
