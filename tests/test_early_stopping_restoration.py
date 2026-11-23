import unittest
import tensorflow as tf
from src.training.early_stopping_restoration import EarlyStoppingWithRestoration

class TestEarlyStoppingRestoration(unittest.TestCase):
    def test_patience_and_stopping(self):
        cb = EarlyStoppingWithRestoration(monitor="val_loss", patience=2, mode="min")
        model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(2,))])
        cb.set_model(model)
        cb.on_train_begin()
        
        # Epoch 0: good
        cb.on_epoch_end(0, {"val_loss": 0.5})
        self.assertEqual(cb.wait, 0)
        # Epoch 1: worse
        cb.on_epoch_end(1, {"val_loss": 0.6})
        self.assertEqual(cb.wait, 1)
        # Epoch 2: worse -> triggers stop
        cb.on_epoch_end(2, {"val_loss": 0.7})
        self.assertTrue(model.stop_training)

if __name__ == '__main__':
    unittest.main()
