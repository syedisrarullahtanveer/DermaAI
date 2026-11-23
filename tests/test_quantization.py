import unittest
import tensorflow as tf
from src.optimization.quantization import ModelQuantizer

class TestQuantization(unittest.TestCase):
    def test_fp16_conversion(self):
        model = tf.keras.Sequential([tf.keras.layers.Dense(4, input_shape=(4,))])
        quantizer = ModelQuantizer()
        tflite_bytes = quantizer.quantize_fp16(model)
        self.assertGreater(len(tflite_bytes), 0)

if __name__ == '__main__':
    unittest.main()
