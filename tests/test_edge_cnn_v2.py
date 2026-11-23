import unittest
import tensorflow as tf
from src.models.edge_cnn_v2 import build_edge_cnn_v2

class TestEdgeCNNv2(unittest.TestCase):
    def test_model_build(self):
        model = build_edge_cnn_v2()
        self.assertEqual(model.output_shape, (None, 7))
        # Run a forward pass with dummy data
        dummy = tf.random.normal([1, 224, 224, 3])
        preds = model(dummy)
        self.assertEqual(preds.shape, (1, 7))

if __name__ == '__main__':
    unittest.main()
