"""Squeeze-and-Excitation (SE) Channel Attention Layer.

Recalibrates channel-wise feature dependencies for convolutional lesion feature maps.

Reference:
    Hu, J., Shen, L., & Sun, G. (2018).
    Squeeze-and-Excitation Networks. In CVPR 2018.
"""

import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class SqueezeAndExcitation2D(layers.Layer):
    """Channel attention layer that models channel-wise interactions via squeeze and excitation."""

    def __init__(self, reduction_ratio: int = 16, **kwargs) -> None:
        """Initialize SE block.

        Args:
            reduction_ratio: Bottleneck reduction ratio for channel squeeze dimension (default 16).
        """
        super().__init__(**kwargs)
        if reduction_ratio < 1:
            raise ValueError(f"reduction_ratio must be >= 1, got {reduction_ratio}")
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(1, channels // self.reduction_ratio)

        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True, name="se_squeeze")
        self.fc1 = layers.Dense(reduced_channels, activation="relu", use_bias=False, name="se_fc1")
        self.fc2 = layers.Dense(channels, activation="sigmoid", use_bias=False, name="se_fc2")
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply Squeeze-and-Excitation attention to feature map tensor."""
        # Squeeze: Global Average Pooling (B, H, W, C) -> (B, 1, 1, C)
        squeezed = self.global_pool(inputs)

        # Excitation: Bottleneck Multi-Layer Perceptron
        excited = self.fc1(squeezed)
        weights = self.fc2(excited)

        # Recalibrate: Broadcast multiplication
        return inputs * weights

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config
