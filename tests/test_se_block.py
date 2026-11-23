"""Unit tests for SqueezeAndExcitation2D module."""

import unittest
import tensorflow as tf
from src.models.se_block import SqueezeAndExcitation2D


class TestSqueezeAndExcitation2D(unittest.TestCase):
    def test_invalid_reduction_ratio_raises(self):
        with self.assertRaises(ValueError):
            SqueezeAndExcitation2D(reduction_ratio=0)

    def test_forward_pass_shape(self):
        layer = SqueezeAndExcitation2D(reduction_ratio=8)
        inputs = tf.random.normal([2, 14, 14, 64])
        outputs = layer(inputs)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_in_functional_model(self):
        inputs = tf.keras.Input(shape=(28, 28, 32))
        x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = SqueezeAndExcitation2D(reduction_ratio=4)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        dummy = tf.random.normal([4, 28, 28, 32])
        preds = model(dummy)
        self.assertEqual(preds.shape, (4, 7))

    def test_serialization(self):
        layer = SqueezeAndExcitation2D(reduction_ratio=12)
        config = layer.get_config()
        self.assertEqual(config["reduction_ratio"], 12)


if __name__ == "__main__":
    unittest.main()
