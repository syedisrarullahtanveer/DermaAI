"""Attention Mechanisms for Skin Lesion Feature Recalibration.

Implements Channel Attention, Spatial Attention, and Convolutional Block Attention Module (CBAM).
"""

import tensorflow as tf
from tensorflow.keras import layers


class ChannelAttention(layers.Layer):
    """Channel Attention Module (CAM).

    Captures 'what' features are clinically salient by modeling inter-channel relationships
    using shared Multi-Layer Perceptron across Average and Max pooled feature descriptors.
    """

    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        reduced_dim = max(channel // self.reduction_ratio, 8)
        self.shared_dense_one = layers.Dense(
            reduced_dim,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True,
            name="cam_dense_1",
        )
        self.shared_dense_two = layers.Dense(
            channel,
            kernel_initializer="he_normal",
            use_bias=True,
            name="cam_dense_2",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Global Average Pooling descriptor
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))

        # Global Max Pooling descriptor
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        # Channel attention map via element-wise addition and sigmoid gating
        scale = tf.sigmoid(avg_out + max_out)
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


class SpatialAttention(layers.Layer):
    """Spatial Attention Module (SAM).

    Captures 'where' features are clinically informative (e.g. lesion perimeter, asymmetry)
    by applying a large receptive field convolution across pooled channel axes.
    """

    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=1,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="sam_conv",
        )
        super().build(input_shape)

    def call(self, inputs):
        # Channel-wise average and max pooling
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # Concatenate along channel dimension
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        scale = self.conv(concat)
        return inputs * scale

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class CBAM(layers.Layer):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies Channel Attention followed by Spatial Attention
    to enable dual-dimensional feature refinement.
    """

    def __init__(self, reduction_ratio: int = 16, spatial_kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size
        self.channel_attention = ChannelAttention(reduction_ratio=reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "spatial_kernel_size": self.spatial_kernel_size,
        })
        return config


class SqueezeAndExcitation(layers.Layer):
    """Squeeze-and-Excitation (SE) Block for adaptive channel-wise feature recalibration."""

    def __init__(self, ratio: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        reduced = max(channel // self.ratio, 4)
        self.fc1 = layers.Dense(reduced, activation="relu", kernel_initializer="he_normal")
        self.fc2 = layers.Dense(channel, activation="sigmoid", kernel_initializer="he_normal")
        super().build(input_shape)

    def call(self, inputs):
        # Squeeze: Global Average Pooling
        squeeze = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)
        # Excitation
        excitation = self.fc2(self.fc1(squeeze))
        excitation = tf.reshape(excitation, [-1, 1, 1, inputs.shape[-1]])
        return inputs * excitation

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config
