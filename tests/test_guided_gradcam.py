"""Unit tests for GuidedGradCAMGenerator."""

import unittest
import numpy as np
import tensorflow as tf
from src.explainability.guided_gradcam import GuidedGradCAMGenerator


class TestGuidedGradCAMGenerator(unittest.TestCase):
    def setUp(self):
        # Create a tiny test model with known conv layer
        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", activation="relu", name="target_conv")(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(3, activation="softmax")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.generator = GuidedGradCAMGenerator(self.model, "target_conv")

    def test_compute_heatmap_shape_and_bounds(self):
        dummy_img = tf.random.normal([1, 32, 32, 3])
        heatmap = self.generator.compute_gradcam_heatmap(dummy_img, target_class_idx=0)
        self.assertEqual(heatmap.shape, (32, 32))
        self.assertTrue(np.all(heatmap >= 0.0) and np.all(heatmap <= 1.0))

    def test_generate_guided_gradcam_output(self):
        dummy_img = tf.random.normal([1, 32, 32, 3])
        guided = self.generator.generate(dummy_img, target_class_idx=1, target_size=(32, 32))
        self.assertEqual(guided.shape, (32, 32, 3))
        self.assertEqual(guided.dtype, np.float32)
        self.assertTrue(np.all(guided >= 0.0) and np.all(guided <= 1.0))


if __name__ == "__main__":
    unittest.main()
