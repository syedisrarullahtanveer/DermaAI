"""Hybrid CNN Architecture with Attention Mechanisms for Skin Lesion Classification."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from src.models.attention import CBAM


def build_hybrid_attention_model(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    backbone_weights: str = "imagenet",
    cbam_reduction_ratio: int = 16,
    dense_units: int = 256,
    dropout_rate: float = 0.35,
    name: str = "DermaAI_Hybrid_CBAM",
) -> tf.keras.Model:
    """Build a hybrid deep model combining EfficientNetB0 feature extraction with CBAM attention.

    Args:
        input_shape: Dimensions of input dermoscopy images.
        num_classes: Target classification count.
        backbone_weights: Pre-trained weights for backbone ('imagenet' or None).
        cbam_reduction_ratio: Bottleneck ratio for channel attention module.
        dense_units: Units in classification projection layer.
        dropout_rate: Dropout rate for regularization.
        name: Model identifier.

    Returns:
        tf.keras.Model: Configured attention-augmented hybrid deep learning model.
    """
    inputs = layers.Input(shape=input_shape, name="image_input")

    # Base backbone extractor
    base_model = EfficientNetB0(
        include_top=False,
        weights=backbone_weights,
        input_shape=input_shape,
    )
    base_model.trainable = False

    # Extract spatial feature map from top convolutional stage
    feature_map = base_model(inputs, training=False)

    # Apply Convolutional Block Attention Module (Channel + Spatial Attention)
    attended_features = CBAM(
        reduction_ratio=cbam_reduction_ratio,
        spatial_kernel_size=7,
        name="cbam_attention_block",
    )(feature_map)

    # Residual attention connection
    refined_features = layers.Add(name="residual_attention_add")([feature_map, attended_features])

    # Global spatial aggregation
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(refined_features)
    x = layers.BatchNormalization(name="bn_post_attention")(x)

    x = layers.Dense(dense_units, activation="relu", kernel_initializer="he_normal", name="dense_proj")(x)
    x = layers.Dropout(dropout_rate, name="head_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model
