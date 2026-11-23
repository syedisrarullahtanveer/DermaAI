import unittest
import tensorflow as tf
from src.models.coordinate_attention import CoordinateAttention

class TestCoordinateAttention(unittest.TestCase):
    def test_forward_shape(self):
        inputs = tf.random.normal([2, 14, 14, 32])
        layer = CoordinateAttention(in_channels=32, reduction=8)
        out = layer(inputs)
        self.assertEqual(out.shape, (2, 14, 14, 32))

if __name__ == '__main__':
    unittest.main()
