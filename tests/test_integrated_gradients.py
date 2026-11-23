"""Unit tests for IntegratedGradientsExplainer."""

import unittest
import numpy as np
import tensorflow as tf
from src.explainability.integrated_gradients import IntegratedGradientsExplainer


class TestIntegratedGradientsExplainer(unittest.TestCase):
    def setUp(self):
        inputs = tf.keras.Input(shape=(16, 16, 3))
        x = tf.keras.layers.Conv2D(4, (3, 3), padding="same", activation="relu")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.explainer = IntegratedGradientsExplainer(self.model, num_steps=5)

    def test_attributions_shape_and_finite(self):
        dummy_img = tf.random.normal([1, 16, 16, 3])
        attrs = self.explainer.generate_attributions(dummy_img, target_class_idx=0)
        self.assertEqual(attrs.shape, (16, 16, 3))
        self.assertTrue(np.all(np.isfinite(attrs)))


if __name__ == "__main__":
    unittest.main()
