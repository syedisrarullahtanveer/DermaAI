import unittest
import numpy as np
import tensorflow as tf
from src.uncertainty.mc_dropout import MCDropoutEstimator

class TestMCDropout(unittest.TestCase):
    def test_mc_dropout_estimation(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation="relu", input_shape=(4,)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(2, activation="softmax")
        ])
        estimator = MCDropoutEstimator(model, num_samples=5)
        x = np.random.randn(2, 4).astype(np.float32)
        res = estimator.predict_with_uncertainty(x)
        self.assertEqual(res["mean"].shape, (2, 2))
        self.assertEqual(res["variance"].shape, (2, 2))

if __name__ == '__main__':
    unittest.main()
