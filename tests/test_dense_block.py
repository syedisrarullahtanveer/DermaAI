import unittest
import tensorflow as tf
from src.models.dense_block import DenseBlock, TransitionLayer

class TestDenseBlock(unittest.TestCase):
    def test_dense_block_forward(self):
        inputs = tf.random.normal([2, 16, 16, 16])
        block = DenseBlock(num_layers=2, growth_rate=8)
        out = block(inputs)
        # 16 + 2 * 8 = 32 channels
        self.assertEqual(out.shape, (2, 16, 16, 32))

    def test_transition_layer(self):
        inputs = tf.random.normal([2, 16, 16, 32])
        trans = TransitionLayer(compression_rate=0.5)
        out = trans(inputs)
        self.assertEqual(out.shape, (2, 8, 8, 16))

if __name__ == '__main__':
    unittest.main()
