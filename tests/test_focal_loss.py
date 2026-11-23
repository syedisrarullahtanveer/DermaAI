"""Unit tests for CategoricalFocalLoss."""

import unittest
import tensorflow as tf
from src.training.focal_loss import CategoricalFocalLoss


class TestCategoricalFocalLoss(unittest.TestCase):
    def test_negative_gamma_raises(self):
        with self.assertRaises(ValueError):
            CategoricalFocalLoss(gamma=-1.0)

    def test_focal_loss_lower_for_confident_prediction(self):
        loss_fn = CategoricalFocalLoss(gamma=2.0)
        y_true = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)

        # Easy confident sample: p = 0.95
        y_pred_easy = tf.constant([[0.95, 0.03, 0.02]], dtype=tf.float32)
        loss_easy = loss_fn(y_true, y_pred_easy).numpy()

        # Hard ambiguous sample: p = 0.40
        y_pred_hard = tf.constant([[0.40, 0.35, 0.25]], dtype=tf.float32)
        loss_hard = loss_fn(y_true, y_pred_hard).numpy()

        self.assertLess(loss_easy, loss_hard)

    def test_gamma_zero_matches_cross_entropy(self):
        focal_loss_fn = CategoricalFocalLoss(gamma=0.0)
        cce_loss_fn = tf.keras.losses.CategoricalCrossentropy()

        y_true = tf.constant([[0.0, 1.0, 0.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.1, 0.8, 0.1]], dtype=tf.float32)

        focal_val = focal_loss_fn(y_true, y_pred).numpy()
        cce_val = cce_loss_fn(y_true, y_pred).numpy()

        self.assertAlmostEqual(focal_val, cce_val, places=4)

    def test_from_logits(self):
        loss_fn = CategoricalFocalLoss(gamma=2.0, from_logits=True)
        y_true = tf.constant([[1.0, 0.0, 0.0]], dtype=tf.float32)
        logits = tf.constant([[3.0, 0.1, -1.0]], dtype=tf.float32)
        val = loss_fn(y_true, logits).numpy()
        self.assertGreater(val, 0.0)


if __name__ == "__main__":
    unittest.main()
