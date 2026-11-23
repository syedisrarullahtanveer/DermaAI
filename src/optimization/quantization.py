"""
Post-training weight quantization utility for edge model deployment.
"""
import tensorflow as tf

class ModelQuantizer:
    def quantize_fp16(self, model: tf.keras.Model) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_quant_model = converter.convert()
        return tflite_quant_model
