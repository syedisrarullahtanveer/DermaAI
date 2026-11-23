"""Unit tests for ModelCalibrationEvaluator."""

import unittest
import numpy as np
from src.evaluation.calibration import ModelCalibrationEvaluator


class TestModelCalibrationEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = ModelCalibrationEvaluator(num_bins=5)

    def test_invalid_num_bins_raises(self):
        with self.assertRaises(ValueError):
            ModelCalibrationEvaluator(num_bins=1)

    def test_perfectly_calibrated_predictions(self):
        # 100 predictions with 1.0 confidence and 100% accuracy
        y_true = np.zeros(100, dtype=int)
        y_prob = np.zeros((100, 2))
        y_prob[:, 0] = 1.0

        metrics = self.evaluator.compute_ece(y_true, y_prob)
        self.assertAlmostEqual(metrics["expected_calibration_error"], 0.0, places=5)
        self.assertAlmostEqual(metrics["maximum_calibration_error"], 0.0, places=5)

    def test_overconfident_predictions_has_positive_ece(self):
        # 100 predictions with 0.99 confidence but 0% accuracy
        y_true = np.ones(100, dtype=int)
        y_prob = np.zeros((100, 2))
        y_prob[:, 0] = 0.99
        y_prob[:, 1] = 0.01

        metrics = self.evaluator.compute_ece(y_true, y_prob)
        self.assertGreater(metrics["expected_calibration_error"], 0.9)
        self.assertEqual(metrics["total_samples"], 100)

    def test_accepts_one_hot_y_true(self):
        y_true = np.array([[1, 0], [0, 1], [1, 0]])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
        metrics = self.evaluator.compute_ece(y_true, y_prob)
        self.assertIn("expected_calibration_error", metrics)
        self.assertEqual(len(metrics["bins"]), 5)


if __name__ == "__main__":
    unittest.main()
