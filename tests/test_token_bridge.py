import unittest
import tensorflow as tf
from src.models.token_bridge import MultiHeadTokenBridge

class TestTokenBridge(unittest.TestCase):
    def test_token_bridge_shape(self):
        inputs = tf.random.normal([2, 8, 8, 16])
        bridge = MultiHeadTokenBridge(embed_dim=32, num_heads=2)
        tokens = bridge(inputs)
        # 8*8 = 64 tokens, embed_dim=32
        self.assertEqual(tokens.shape, (2, 64, 32))

if __name__ == '__main__':
    unittest.main()
