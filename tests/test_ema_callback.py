"""Unit tests for ExponentialMovingAverageCallback."""

import unittest
import numpy as np
import tensorflow as tf
from src.training.ema_callback import ExponentialMovingAverageCallback


class TestExponentialMovingAverageCallback(unittest.TestCase):
    def test_ema_weight_averaging(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(2, input_shape=(4,), use_bias=False)
        ])
        # Set initial weight to 1.0
        model.set_weights([np.ones((4, 2), dtype=np.float32)])

        callback = ExponentialMovingAverageCallback(decay=0.5)
        callback.set_model(model)
        callback.on_train_begin()

        # Update model weights to 3.0
        model.set_weights([np.full((4, 2), 3.0, dtype=np.float32)])
        callback.on_batch_end(batch=0)

        # Expected EMA: 0.5 * 1.0 + 0.5 * 3.0 = 2.0
        self.assertAlmostEqual(float(callback.ema_weights[0][0, 0]), 2.0, places=4)

        # Apply EMA back to model
        callback.apply_ema_weights()
        self.assertAlmostEqual(float(model.get_weights()[0][0, 0]), 2.0, places=4)


if __name__ == "__main__":
    unittest.main()
