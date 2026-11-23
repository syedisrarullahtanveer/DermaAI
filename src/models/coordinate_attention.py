"""
Coordinate Attention for mobile and edge dermoscopy architectures.
Reference: Hou et al., "Coordinate Attention for Efficient Mobile Network Design", CVPR 2021.
"""
import tensorflow as tf
from keras import layers

class CoordinateAttention(layers.Layer):
    def __init__(self, in_channels: int, reduction: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.reduction = reduction
        self.reduced_channels = max(1, in_channels // reduction)

    def build(self, input_shape):
        self.conv1 = layers.Conv2D(self.reduced_channels, 1, use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv_h = layers.Conv2D(self.in_channels, 1, activation="sigmoid", use_bias=False)
        self.conv_w = layers.Conv2D(self.in_channels, 1, activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, x, training=False):
        h, w = x.shape[1], x.shape[2]
        # Pool across spatial dimensions separately
        x_h = tf.reduce_mean(x, axis=2, keepdims=True) # (N, H, 1, C)
        x_w = tf.reduce_mean(x, axis=1, keepdims=True) # (N, 1, W, C)
        x_w = tf.transpose(x_w, [0, 2, 1, 3])          # (N, W, 1, C)
        
        y = tf.concat([x_h, x_w], axis=1)              # (N, H+W, 1, C)
        y = tf.nn.silu(self.bn1(self.conv1(y), training=training))
        
        y_h, y_w = tf.split(y, [h, w], axis=1)
        y_w = tf.transpose(y_w, [0, 2, 1, 3])
        
        a_h = self.conv_h(y_h)
        a_w = self.conv_w(y_w)
        return x * a_h * a_w
