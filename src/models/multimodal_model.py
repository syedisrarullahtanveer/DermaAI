"""Multimodal Fusion Architecture Combining Dermoscopy Images and Patient Metadata."""

from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_multimodal_fusion_model(
    image_shape: Tuple[int, int, int] = (224, 224, 3),
    metadata_dim: int = 15,
    num_classes: int = 7,
    image_feature_dim: int = 256,
    fusion_type: str = "gated",
    dropout_rate: float = 0.3,
    name: str = "DermaAI_Multimodal_Fusion",
) -> tf.keras.Model:
    """Build a dual-branch network fusing high-resolution dermoscopic image features with tabular demographic records.

    Args:
        image_shape: Input image dimensions (H, W, C).
        metadata_dim: Dimension of preprocessed clinical tabular feature vector.
        num_classes: Target classification count.
        image_feature_dim: Dimension of projected visual feature embedding.
        fusion_type: Fusion strategy ('concatenation', 'gated').
        dropout_rate: Regularization dropout rate.
        name: Model identifier.

    Returns:
        tf.keras.Model: Multimodal deep learning model.
    """
    # Branch 1: Dermoscopic Image Vision Backbone
    image_inputs = layers.Input(shape=image_shape, name="image_input")
    base_vision = EfficientNetB0(include_top=False, weights="imagenet", input_shape=image_shape)
    base_vision.trainable = False

    x_img = base_vision(image_inputs, training=False)
    x_img = layers.GlobalAveragePooling2D(name="img_global_pool")(x_img)
    x_img = layers.Dense(image_feature_dim, activation="relu", name="img_dense_embedding")(x_img)
    x_img = layers.BatchNormalization(name="img_embedding_bn")(x_img)

    # Branch 2: Tabular Clinical Demographic MLP
    meta_inputs = layers.Input(shape=(metadata_dim,), name="metadata_input")
    x_meta = layers.Dense(64, activation="relu", name="meta_dense_1")(meta_inputs)
    x_meta = layers.BatchNormalization(name="meta_bn_1")(x_meta)
    x_meta = layers.Dropout(0.2, name="meta_dropout_1")(x_meta)
    x_meta = layers.Dense(image_feature_dim, activation="relu", name="meta_dense_embedding")(x_meta)
    x_meta = layers.BatchNormalization(name="meta_embedding_bn")(x_meta)

    # Multimodal Fusion Strategy
    if fusion_type == "gated":
        # Gated Multimodal Unit: learns dynamic weighting between vision and clinical metadata
        gate = layers.Dense(image_feature_dim, activation="sigmoid", name="multimodal_gate")(
            layers.Concatenate()([x_img, x_meta])
        )
        fused = layers.Add(name="gated_fusion")([
            layers.Multiply()([gate, x_img]),
            layers.Multiply()([1.0 - gate, x_meta]),
        ])
    else:
        # Standard early concatenation
        fused = layers.Concatenate(name="concat_fusion")([x_img, x_meta])

    # Final Classification Head
    x = layers.Dense(128, activation="relu", kernel_initializer="he_normal", name="fused_dense")(fused)
    x = layers.Dropout(dropout_rate, name="fused_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output",
    )(x)

    model = models.Model(inputs=[image_inputs, meta_inputs], outputs=outputs, name=name)
    return model
