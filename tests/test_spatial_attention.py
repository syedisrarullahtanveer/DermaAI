"""Unit tests for SpatialAttention2D module."""

import unittest
import tensorflow as tf
from src.models.spatial_attention import SpatialAttention2D


class TestSpatialAttention2D(unittest.TestCase):
    def test_even_kernel_raises_value_error(self):
        with self.assertRaises(ValueError):
            SpatialAttention2D(kernel_size=4)

    def test_layer_forward_pass_shape(self):
        layer = SpatialAttention2D(kernel_size=7)
        dummy_input = tf.random.normal([2, 16, 16, 32])
        output = layer(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_layer_in_keras_model(self):
        inputs = tf.keras.Input(shape=(32, 32, 64))
        x = SpatialAttention2D(kernel_size=5)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        dummy = tf.random.normal([4, 32, 32, 64])
        preds = model(dummy)
        self.assertEqual(preds.shape, (4, 7))

    def test_serialization(self):
        layer = SpatialAttention2D(kernel_size=3)
        config = layer.get_config()
        self.assertEqual(config["kernel_size"], 3)


if __name__ == "__main__":
    unittest.main()
