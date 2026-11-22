"""Attention Mechanisms for Skin Lesion Feature Recalibration.

Implements Channel Attention, Spatial Attention, CBAM (Convolutional Block Attention Module),
and Squeeze-and-Excitation (SE) for adaptive feature refinement in dermoscopy image analysis.

Mathematical Formulations:
-------------------------
1. Channel Attention Module (CAM):
   M_c(F) = σ( MLP(AvgPool(F)) + MLP(MaxPool(F)) )
          = σ( W_1(δ(W_0(F_avg^c))) + W_1(δ(W_0(F_max^c))) )
   where:
     F ∈ R^{H × W × C}: Input intermediate feature map.
     AvgPool(F), MaxPool(F) ∈ R^{1 × 1 × C}: Spatial pooling descriptors.
     W_0 ∈ R^{(C/r) × C}, W_1 ∈ R^{C × (C/r)}: Shared MLP bottleneck weights with reduction ratio r.
     δ: ReLU activation function.
     σ: Sigmoid activation function.

2. Spatial Attention Module (SAM):
   M_s(F) = σ( f^{k × k}( [AvgPool_c(F); MaxPool_c(F)] ) )
   where:
     AvgPool_c(F), MaxPool_c(F) ∈ R^{H × W × 1}: Channel-wise pooling descriptors.
     [ ; ]: Concatenation along the channel axis.
     f^{k × k}: Convolutional operation with kernel size k × k (typically 7 × 7).
     σ: Sigmoid activation function.

3. Convolutional Block Attention Module (CBAM):
   F' = M_c(F) ⊗ F
   F'' = M_s(F') ⊗ F'
   F_out = F + F''  (with residual identity shortcut)
   where ⊗ denotes element-wise broadcast multiplication.
"""

from typing import Any, Dict, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class ChannelAttention(layers.Layer):
    """Channel Attention Module (CAM).

    Recalibrates channel-wise feature responses by explicitly modeling
    interdependencies between feature channels. Utilizes both Average-Pooling
    and Max-Pooling to capture comprehensive spatial statistics.

    Args:
        reduction_ratio: Dimensionality reduction ratio in the shared MLP bottleneck.
        name: Optional layer name.
    """

    def __init__(self, reduction_ratio: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = max(1, int(reduction_ratio))

    def build(self, input_shape: tf.TensorShape) -> None:
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

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Spatial pooling descriptors: (B, 1, 1, C)
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        # Forward descriptors through shared two-layer bottleneck MLP
        avg_out = self.shared_dense_two(self.shared_dense_one(avg_pool))
        max_out = self.shared_dense_two(self.shared_dense_one(max_pool))

        # Channel attention map via element-wise addition and Sigmoid gating
        channel_attention = tf.sigmoid(avg_out + max_out)

        # Modulate original features by channel attention weights
        return inputs * channel_attention

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class SpatialAttention(layers.Layer):
    """Spatial Attention Module (SAM).

    Captures inter-spatial feature relationships by concentrating on 'where'
    informative lesions, borders, and pigment networks reside. Employs channel-wise
    pooling descriptors combined with large receptive field convolution.

    Args:
        kernel_size: Convolutional kernel size (e.g. 7 for large spatial context).
        name: Optional layer name.
    """

    def __init__(self, kernel_size: int = 7, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.conv = layers.Conv2D(
            filters=1,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=1,
            padding="same",
            activation="sigmoid",
            use_bias=False,
            kernel_initializer="he_normal",
            name="sam_conv",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Channel-wise pooling descriptors: (B, H, W, 1)
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # Concatenate along channel dimension: (B, H, W, 2)
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Compute spatial attention map via 2D convolution: (B, H, W, 1)
        spatial_attention = self.conv(concat)

        # Modulate feature map
        return inputs * spatial_attention

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class CBAM(layers.Layer):
    """Convolutional Block Attention Module (CBAM).

    Sequentially infers attention maps along two separate dimensions (Channel then Spatial),
    then multiplies the attention maps to the input feature map for adaptive feature refinement.
    Optionally adds a residual shortcut connection.

    Args:
        reduction_ratio: Channel attention bottleneck reduction ratio.
        spatial_kernel_size: Spatial attention convolution kernel size.
        use_residual: Whether to add a residual identity connection F + F''.
        name: Optional layer name.
    """

    def __init__(
        self,
        reduction_ratio: int = 16,
        spatial_kernel_size: int = 7,
        use_residual: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        self.spatial_kernel_size = spatial_kernel_size
        self.use_residual = use_residual

        self.channel_attention = ChannelAttention(
            reduction_ratio=reduction_ratio,
            name=f"{self.name}_channel_attention" if self.name else None,
        )
        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel_size,
            name=f"{self.name}_spatial_attention" if self.name else None,
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Step 1: Channel recalibration
        x_channel = self.channel_attention(inputs)

        # Step 2: Spatial recalibration
        x_attended = self.spatial_attention(x_channel)

        # Optional residual identity connection
        if self.use_residual:
            return inputs + x_attended
        return x_attended

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({
            "reduction_ratio": self.reduction_ratio,
            "spatial_kernel_size": self.spatial_kernel_size,
            "use_residual": self.use_residual,
        })
        return config


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class SqueezeAndExcitation(layers.Layer):
    """Squeeze-and-Excitation (SE) Block.

    Adaptively recalibrates channel-wise feature responses by explicitly modeling
    channel interdependencies using global average pooling and a two-stage FC gating.

    Args:
        ratio: Dimensionality reduction ratio.
        name: Optional layer name.
    """

    def __init__(self, ratio: int = 16, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.ratio = max(1, int(ratio))

    def build(self, input_shape: tf.TensorShape) -> None:
        channel = input_shape[-1]
        reduced = max(channel // self.ratio, 4)

        self.fc1 = layers.Dense(
            reduced,
            activation="relu",
            kernel_initializer="he_normal",
            name="se_dense_1",
        )
        self.fc2 = layers.Dense(
            channel,
            activation="sigmoid",
            kernel_initializer="he_normal",
            name="se_dense_2",
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # Squeeze: Global Average Pooling across spatial dimensions
        squeeze = tf.reduce_mean(inputs, axis=[1, 2], keepdims=False)

        # Excitation: Two-layer MLP gating
        excitation = self.fc2(self.fc1(squeeze))

        # Reshape to (B, 1, 1, C) for broadcasting
        excitation_expanded = tf.reshape(excitation, [-1, 1, 1, tf.shape(inputs)[-1]])
        return inputs * excitation_expanded

    def compute_output_shape(self, input_shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        return input_shape

    def get_config(self) -> Dict[str, Any]:
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config
