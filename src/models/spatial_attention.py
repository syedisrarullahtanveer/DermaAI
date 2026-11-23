"""Spatial Attention Module for Deep Learning Dermatological Feature Extractors.

Reference:
    Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018).
    CBAM: Convolutional Block Attention Module. In ECCV 2018.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class SpatialAttention2D(layers.Layer):
    """Spatial attention layer that computes a 2D spatial saliency mask over feature maps."""

    def __init__(self, kernel_size: int = 7, **kwargs):
        """Initialize SpatialAttention2D.

        Args:
            kernel_size: Convolution kernel size for inter-spatial relationship modeling (default 7).
        """
        super().__init__(**kwargs)
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got: {kernel_size}")
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            name="spatial_attention_conv",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Compute spatial attention map and apply it element-wise to inputs.

        Args:
            inputs: Tensor of shape (batch, height, width, channels).

        Returns:
            tf.Tensor: Attended feature maps of the same shape as inputs.
        """
        # Average pooling across channel dimension: shape (batch, height, width, 1)
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pooling across channel dimension: shape (batch, height, width, 1)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # Concatenate along channel axis: shape (batch, height, width, 2)
        concat = layers.concatenate([avg_pool, max_pool], axis=-1)

        # Generate spatial attention mask: shape (batch, height, width, 1)
        attention_map = self.conv(concat)

        # Scale inputs with attention mask
        return inputs * attention_map

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config
