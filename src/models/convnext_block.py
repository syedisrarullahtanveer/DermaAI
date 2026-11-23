"""ConvNeXt-inspired Large-Kernel Depthwise Separable Block.

Reference:
    Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
    A ConvNet for the 2020s. In CVPR 2022.
"""

from typing import Dict, Any
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class ConvNeXtBlock(layers.Layer):
    """ConvNeXt block with 7x7 depthwise convolution, LayerNorm, inverted bottleneck, and GELU."""

    def __init__(self, dim: int, expansion: int = 4, **kwargs) -> None:
        """Initialize ConvNeXt block.

        Args:
            dim: Input and output channel dimension.
            expansion: Expansion factor for inverted bottleneck MLP (default 4).
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.expansion = expansion

        # 7x7 Depthwise Convolution
        self.dwconv = layers.DepthwiseConv2D(kernel_size=7, padding="same", name="dwconv")
        # Layer Normalization across channels
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="layernorm")
        # Inverted bottleneck point-wise convolutions with GELU
        self.pwconv1 = layers.Dense(expansion * dim, activation="gelu", name="pwconv1")
        self.pwconv2 = layers.Dense(dim, name="pwconv2")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply ConvNeXt residual transform."""
        residual = inputs
        x = self.dwconv(inputs)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.pwconv2(x)
        return residual + x

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "expansion": self.expansion,
        })
        return config
