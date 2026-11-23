import unittest
import tensorflow as tf
from src.training.label_smoothing import LabelSmoothingCrossEntropy

class TestLabelSmoothing(unittest.TestCase):
    def test_loss_computation(self):
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        y_true = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        y_pred = tf.constant([[0.9, 0.1], [0.1, 0.9]])
        loss = loss_fn(y_true, y_pred)
        self.assertGreater(float(loss), 0.0)
        self.assertFalse(tf.math.is_nan(loss))

    def test_invalid_smoothing(self):
        with self.assertRaises(ValueError):
            LabelSmoothingCrossEntropy(smoothing=1.5)

if __name__ == '__main__':
    unittest.main()
