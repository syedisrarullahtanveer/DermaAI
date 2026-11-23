"""Unit tests for MulticlassROCEvaluator."""

import unittest
import numpy as np
from src.evaluation.roc_metrics import MulticlassROCEvaluator


class TestMulticlassROCEvaluator(unittest.TestCase):
    def setUp(self):
        self.class_names = ["MEL", "NV", "BCC"]
        self.evaluator = MulticlassROCEvaluator(class_names=self.class_names)

    def test_perfect_predictions_yield_auc_one(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # Perfect probabilities
        y_score = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.85, 0.05],
            [0.05, 0.1, 0.85],
            [0.8, 0.1, 0.1],
            [0.15, 0.75, 0.1],
            [0.1, 0.1, 0.8],
        ])

        results = self.evaluator.evaluate(y_true, y_score)
        self.assertAlmostEqual(results["macro_auc"], 1.0, places=4)
        self.assertAlmostEqual(results["weighted_auc"], 1.0, places=4)
        for name in self.class_names:
            self.assertAlmostEqual(results["per_class_auc"][name], 1.0, places=4)

    def test_accepts_one_hot_targets(self):
        y_true = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        y_score = np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
        ])
        results = self.evaluator.evaluate(y_true, y_score)
        self.assertEqual(results["num_classes"], 3)
        self.assertIn("macro_auc", results)


if __name__ == "__main__":
    unittest.main()
