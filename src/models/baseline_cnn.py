"""Modular Baseline Deep Convolutional Neural Network for Skin Lesion Classification."""

from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    conv_filters: List[int] = None,
    dense_units: List[int] = None,
    dropout_rate: float = 0.3,
    l2_reg: float = 0.0,
    use_batch_norm: bool = True,
    name: str = "DermaCNN_Baseline",
) -> tf.keras.Model:
    """Build a modular custom deep CNN with batch normalization, residual skip blocks, and spatial regularization.

    Args:
        input_shape: Tuple specifying input image dimensions (H, W, C).
        num_classes: Number of lesion classes (default: 7 for HAM10000).
        conv_filters: List of filter counts for progressive convolutional stages.
        dense_units: List of unit counts for dense classification head.
        dropout_rate: Dropout rate for regularization.
        l2_reg: L2 kernel regularization factor.
        use_batch_norm: Whether to apply BatchNormalization after convolutions.
        name: Model name.

    Returns:
        tf.keras.Model: Compiled or uncompiled Keras CNN model.
    """
    if conv_filters is None:
        conv_filters = [32, 64, 128, 256]
    if dense_units is None:
        dense_units = [256, 128]

    regularizer = tf.keras.regularizers.l2(l2_reg) if l2_reg > 0 else None

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = inputs

    # Progressive Convolutional Feature Extractor
    for i, filters in enumerate(conv_filters):
        # First conv in stage
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
            name=f"conv_stage_{i+1}_1",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_stage_{i+1}_1")(x)
        x = layers.ReLU(name=f"relu_stage_{i+1}_1")(x)

        # Second conv in stage with skip-like feature accumulation
        x = layers.Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            padding="same",
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
            name=f"conv_stage_{i+1}_2",
        )(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_stage_{i+1}_2")(x)
        x = layers.ReLU(name=f"relu_stage_{i+1}_2")(x)

        # Spatial downsampling
        x = layers.MaxPooling2D(pool_size=(2, 2), name=f"pool_stage_{i+1}")(x)
        x = layers.SpatialDropout2D(rate=min(0.1 * (i + 1), 0.3), name=f"spatial_dropout_{i+1}")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    for j, units in enumerate(dense_units):
        x = layers.Dense(units=units, kernel_initializer="he_normal", name=f"dense_{j+1}")(x)
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"bn_dense_{j+1}")(x)
        x = layers.ReLU(name=f"relu_dense_{j+1}")(x)
        x = layers.Dropout(rate=dropout_rate, name=f"dropout_dense_{j+1}")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model
