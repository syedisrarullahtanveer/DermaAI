"""
Spatial Pyramid Pooling (SPP) layer for multi-scale dermoscopic feature capture.
Reference: He et al., "Spatial Pyramid Pooling in Deep Convolutional Networks", TPAMI 2015.
"""
import tensorflow as tf
from keras import layers

class SpatialPyramidPooling2D(layers.Layer):
    """Pool feature maps across multiple spatial bins and concatenate.

    Args:
        bin_sizes: List of integers specifying the grid divisions (e.g. [1, 2, 4]).
    """
    def __init__(self, bin_sizes=(1, 2, 4), **kwargs):
        super().__init__(**kwargs)
        self.bin_sizes = list(bin_sizes)

    def call(self, inputs):
        h, w = inputs.shape[1], inputs.shape[2]
        pooled_outputs = []
        for b in self.bin_sizes:
            pool_h = max(1, h // b)
            pool_w = max(1, w // b)
            p = layers.MaxPooling2D(pool_size=(pool_h, pool_w), strides=(pool_h, pool_w), padding='valid')(inputs)
            f = layers.Flatten()(p)
            pooled_outputs.append(f)
        return tf.concat(pooled_outputs, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"bin_sizes": self.bin_sizes})
        return cfg
