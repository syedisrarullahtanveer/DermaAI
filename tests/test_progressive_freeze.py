import unittest
import tensorflow as tf
from src.training.progressive_freeze import ProgressiveFreezeCallback

class TestProgressiveFreeze(unittest.TestCase):
    def test_unfreezing_schedule(self):
        l1 = tf.keras.layers.Dense(4)
        l2 = tf.keras.layers.Dense(2)
        l1.trainable = False
        l2.trainable = False
        model = tf.keras.Sequential([l1, l2])
        
        cb = ProgressiveFreezeCallback(unfreeze_schedule={1: 1})
        cb.set_model(model)
        cb.on_epoch_begin(1)
        self.assertTrue(l2.trainable)
        self.assertFalse(l1.trainable)

if __name__ == '__main__':
    unittest.main()
