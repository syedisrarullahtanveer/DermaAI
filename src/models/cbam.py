"""
Convolutional Block Attention Module (CBAM) for dermoscopic lesion feature recalibration.
Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
"""
import tensorflow as tf
from keras import layers

class ChannelAttention(layers.Layer):
    def __init__(self, in_planes: int, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.shared_mlp_1 = layers.Dense(max(1, in_planes // ratio), activation="relu", use_bias=False)
        self.shared_mlp_2 = layers.Dense(in_planes, use_bias=False)

    def call(self, x):
        avg_out = self.shared_mlp_2(self.shared_mlp_1(self.avg_pool(x)))
        max_out = self.shared_mlp_2(self.shared_mlp_1(self.max_pool(x)))
        scale = tf.sigmoid(avg_out + max_out)
        return x * scale

class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size, padding="same", activation="sigmoid", use_bias=False)

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        scale = self.conv(concat)
        return x * scale

class CBAM(layers.Layer):
    def __init__(self, in_planes: int, ratio: int = 8, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def call(self, x):
        return self.sa(self.ca(x))
