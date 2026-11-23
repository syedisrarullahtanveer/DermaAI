import unittest
import tensorflow as tf
from src.training.poly_loss import PolyLoss

class TestPolyLoss(unittest.TestCase):
    def test_loss_value(self):
        loss_fn = PolyLoss(epsilon=1.0)
        y_t = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        y_p = tf.constant([[0.9, 0.1], [0.2, 0.8]])
        val = loss_fn(y_t, y_p)
        self.assertGreater(float(val), 0.0)

if __name__ == '__main__':
    unittest.main()
