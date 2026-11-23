"""
Inverted Residual Block (MobileNetV3 style) for mobile dermoscopy deployment.
"""
import tensorflow as tf
from keras import layers

class InvertedResidualBlock(layers.Layer):
    def __init__(self, in_channels: int, out_channels: int, expansion_factor: int = 3, stride: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.use_residual = stride == 1 and in_channels == out_channels
        hidden_dim = in_channels * expansion_factor
        
        self.expand = layers.Conv2D(hidden_dim, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.dwconv = layers.DepthwiseConv2D(3, strides=stride, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.project = layers.Conv2D(out_channels, 1, use_bias=False)
        self.bn3 = layers.BatchNormalization()

    def call(self, x, training=False):
        h = tf.nn.silu(self.bn1(self.expand(x), training=training))
        h = tf.nn.silu(self.bn2(self.dwconv(h), training=training))
        h = self.bn3(self.project(h), training=training)
        return x + h if self.use_residual else h
