"""
DenseNet-inspired dense connectivity block for DermaAI.

Each layer receives feature maps from all preceding layers,
improving gradient flow and feature reuse.
"""

import tensorflow as tf
from keras import layers


class DenseLayer(layers.Layer):
    """Single dense layer: BN -> ReLU -> Conv2D."""

    def __init__(self, growth_rate: int = 32, bottleneck_width: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.growth_rate = growth_rate
        bottleneck_filters = bottleneck_width * growth_rate
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(bottleneck_filters, 1, use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False)

    def call(self, x, training=False):
        h = self.conv1(tf.nn.relu(self.bn1(x, training=training)))
        h = self.conv2(tf.nn.relu(self.bn2(h, training=training)))
        return tf.concat([x, h], axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"growth_rate": self.growth_rate})
        return cfg


class DenseBlock(layers.Layer):
    """Stack of DenseLayers with feature map concatenation.

    Args:
        num_layers: Number of dense layers in the block.
        growth_rate: Number of feature maps added per layer.
    """

    def __init__(self, num_layers: int = 4, growth_rate: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.dense_layers = [DenseLayer(growth_rate) for _ in range(num_layers)]

    def call(self, x, training=False):
        for layer in self.dense_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_layers": self.num_layers, "growth_rate": self.growth_rate})
        return cfg


class TransitionLayer(layers.Layer):
    """Compression transition layer between dense blocks.

    Reduces feature map dimensions by compression_rate.
    """

    def __init__(self, compression_rate: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.compression_rate = compression_rate
        self.bn = layers.BatchNormalization()
        self._conv = None  # built lazily
        self.pool = layers.AveragePooling2D(pool_size=2, strides=2)

    def build(self, input_shape):
        in_filters = input_shape[-1]
        out_filters = max(1, int(in_filters * self.compression_rate))
        self._conv = layers.Conv2D(out_filters, 1, use_bias=False)
        super().build(input_shape)

    def call(self, x, training=False):
        x = self._conv(tf.nn.relu(self.bn(x, training=training)))
        return self.pool(x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"compression_rate": self.compression_rate})
        return cfg
