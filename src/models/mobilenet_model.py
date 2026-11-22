"""MobileNetV2 Transfer Learning Model for Skin Lesion Classification.

MobileNetV2 uses depthwise separable convolutions for efficient inference,
making it well-suited for resource-constrained clinical deployment while
still achieving competitive diagnostic accuracy via transfer learning.
"""

from typing import List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


def build_mobilenet(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    weights: str = "imagenet",
    freeze_base: bool = True,
    dense_units: List[int] = None,
    dropout_rate: float = 0.25,
    name: str = "MobileNetV2_Derma",
) -> tf.keras.Model:
    """Build MobileNetV2-based transfer learning model for dermoscopy classification.

    MobileNetV2 requires pixel values in [-1, 1]. The built-in preprocessing
    function (`preprocess_input`) is embedded in the graph, so no external
    normalization is required at inference time.

    Args:
        input_shape: Input image shape (H, W, C). Recommended: (224, 224, 3).
        num_classes: Target lesion class count (7 for HAM10000).
        weights: 'imagenet' or None.
        freeze_base: Freeze backbone for stage-1 feature extraction.
        dense_units: Dense units in the classification head.
        dropout_rate: Dropout rate in the classification neck.
        name: Keras model name.

    Returns:
        tf.keras.Model: Configured MobileNetV2 model ready for compilation.
    """
    if dense_units is None:
        dense_units = [256]

    # MobileNetV2 backbone
    base_model = MobileNetV2(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        alpha=1.0,  # Width multiplier: 1.0 = full capacity
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="image_input")

    # MobileNetV2 expects [-1, 1] normalization
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=not freeze_base)

    # Spatial aggregation
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn_0")(x)

    # Classification head
    for idx, units in enumerate(dense_units):
        x = layers.Dense(
            units,
            kernel_initializer="he_normal",
            use_bias=False,
            name=f"head_dense_{idx + 1}",
        )(x)
        x = layers.BatchNormalization(name=f"head_bn_{idx + 1}")(x)
        x = layers.ReLU(name=f"head_relu_{idx + 1}")(x)
        x = layers.Dropout(dropout_rate, name=f"head_dropout_{idx + 1}")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def unfreeze_mobilenet_top(model: tf.keras.Model, num_layers: int = 20) -> None:
    """Unfreeze the top-N layers of the MobileNetV2 backbone for fine-tuning.

    Args:
        model: Compiled MobileNetV2 Keras model.
        num_layers: Number of top layers to unfreeze.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("MobileNetV2 base model layer not found.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers)

    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[MobileNetV2] Unfrozen {n_trainable}/{total_layers} backbone layers for fine-tuning.")
