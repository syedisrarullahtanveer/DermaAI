import unittest
import tensorflow as tf
from src.training.stochastic_depth import StochasticDepth, StochasticDepthCallback

class TestStochasticDepth(unittest.TestCase):
    def test_eval_mode_identity(self):
        layer = StochasticDepth(drop_rate=0.5)
        x = tf.ones([2, 4, 4, 8])
        res = tf.ones([2, 4, 4, 8])
        out = layer(x, res, training=False)
        self.assertTrue(tf.reduce_all(tf.equal(out, 2.0)))

    def test_callback_schedule(self):
        layer = StochasticDepth(drop_rate=0.0)
        cb = StochasticDepthCallback([layer], max_drop_rate=0.4)
        cb.params = {"epochs": 5}
        cb.on_train_begin()
        cb.on_epoch_begin(4)
        self.assertAlmostEqual(layer.drop_rate, 0.4, places=2)

if __name__ == '__main__':
    unittest.main()
