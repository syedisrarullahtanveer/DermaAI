"""Unit tests for ClinicalMetricsSummary."""

import unittest
import numpy as np
from src.evaluation.confusion_matrix import ClinicalMetricsSummary


class TestClinicalMetricsSummary(unittest.TestCase):
    def setUp(self):
        self.class_names = ["MEL", "NV"]
        self.analyzer = ClinicalMetricsSummary(class_names=self.class_names)

    def test_metrics_calculation(self):
        # 2 MEL (one classified as MEL, one as NV)
        # 2 NV (both classified as NV)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        res = self.analyzer.compute_summary(y_true, y_pred)
        self.assertEqual(res["total_samples"], 4)
        self.assertAlmostEqual(res["overall_accuracy"], 0.75)

        mel_metrics = res["per_class_metrics"]["MEL"]
        self.assertEqual(mel_metrics["tp"], 1)
        self.assertEqual(mel_metrics["fn"], 1)
        self.assertEqual(mel_metrics["fp"], 0)
        self.assertEqual(mel_metrics["tn"], 2)
        self.assertAlmostEqual(mel_metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(mel_metrics["specificity"], 1.0)
        self.assertAlmostEqual(mel_metrics["ppv"], 1.0)


if __name__ == "__main__":
    unittest.main()
