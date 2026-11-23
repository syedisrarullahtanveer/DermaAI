import unittest
import tensorflow as tf
from src.models.inverted_residual import InvertedResidualBlock

class TestInvertedResidual(unittest.TestCase):
    def test_residual_stride_1(self):
        inputs = tf.random.normal([2, 16, 16, 16])
        block = InvertedResidualBlock(in_channels=16, out_channels=16, stride=1)
        out = block(inputs)
        self.assertEqual(out.shape, (2, 16, 16, 16))

    def test_downsample_stride_2(self):
        inputs = tf.random.normal([2, 16, 16, 16])
        block = InvertedResidualBlock(in_channels=16, out_channels=32, stride=2)
        out = block(inputs)
        self.assertEqual(out.shape, (2, 8, 8, 32))

if __name__ == '__main__':
    unittest.main()
