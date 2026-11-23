"""Unit tests for ConvNeXtBlock."""

import unittest
import tensorflow as tf
from src.models.convnext_block import ConvNeXtBlock


class TestConvNeXtBlock(unittest.TestCase):
    def test_forward_pass_shape(self):
        block = ConvNeXtBlock(dim=32, expansion=4)
        inputs = tf.random.normal([2, 16, 16, 32])
        outputs = block(inputs)
        self.assertEqual(outputs.shape, inputs.shape)

    def test_in_sequential_or_functional_model(self):
        inputs = tf.keras.Input(shape=(32, 32, 16))
        x = ConvNeXtBlock(dim=16, expansion=2)(inputs)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        outputs = tf.keras.layers.Dense(7, activation="softmax")(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        dummy = tf.random.normal([4, 32, 32, 16])
        preds = model(dummy)
        self.assertEqual(preds.shape, (4, 7))

    def test_serialization(self):
        block = ConvNeXtBlock(dim=48, expansion=3)
        config = block.get_config()
        self.assertEqual(config["dim"], 48)
        self.assertEqual(config["expansion"], 3)


if __name__ == "__main__":
    unittest.main()
