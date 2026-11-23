"""Unit tests for GradCAMPlusPlus."""

import unittest
import numpy as np
import tensorflow as tf
from src.explainability.gradcam_plus_plus import GradCAMPlusPlus


class TestGradCAMPlusPlus(unittest.TestCase):
    def setUp(self):
        inputs = tf.keras.Input(shape=(28, 28, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu", name="target_conv")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.explainer = GradCAMPlusPlus(self.model, "target_conv")

    def test_heatmap_shape_and_range(self):
        dummy_img = tf.random.normal([1, 28, 28, 3])
        heatmap = self.explainer.generate_heatmap(dummy_img, target_class_idx=1)
        self.assertEqual(heatmap.shape, (28, 28))
        self.assertTrue(np.all(heatmap >= 0.0))
        self.assertTrue(np.all(heatmap <= 1.0))


if __name__ == "__main__":
    unittest.main()
