"""ResNet50 Transfer Learning Model for Skin Lesion Classification.

Implements two-stage training:
  Stage 1 — Feature extraction with frozen backbone (top classification head only).
  Stage 2 — Fine-tuning with top-N layers unfrozen at a lower learning rate.
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


def build_resnet50(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    weights: str = "imagenet",
    freeze_base: bool = True,
    dense_units: List[int] = None,
    dropout_rate: float = 0.3,
    name: str = "ResNet50_Derma",
) -> tf.keras.Model:
    """Build ResNet50-based transfer learning model for dermoscopy classification.

    Args:
        input_shape: Input image dimensions (H, W, C). Must be >= (32, 32, 3).
        num_classes: Number of skin lesion classes (7 for HAM10000).
        weights: Pre-trained weights source ('imagenet' or None for random init).
        freeze_base: If True, backbone is frozen for feature-extraction stage.
        dense_units: Dense layer units in the classification head.
        dropout_rate: Dropout rate applied in the classification neck.
        name: Model name identifier.

    Returns:
        tf.keras.Model: Configured ResNet50 model, ready for compilation.
    """
    if dense_units is None:
        dense_units = [512, 256]

    # Base model: ResNet50 without top classification layers
    base_model = ResNet50(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling=None,
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="image_input")

    # ImageNet preprocessing built into the graph for portability
    x = tf.keras.applications.resnet50.preprocess_input(inputs)
    x = base_model(x, training=not freeze_base)

    # Spatial feature aggregation
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="head_bn_0")(x)

    # Classification head with progressive compression
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


def unfreeze_resnet50_top(model: tf.keras.Model, num_layers: int = 25) -> None:
    """Unfreeze the top-N layers of the ResNet50 backbone for fine-tuning.

    Keeps BatchNormalization layers frozen to preserve running statistics
    learned during ImageNet pre-training.

    Args:
        model: Compiled ResNet50 Keras model.
        num_layers: Number of top layers to unfreeze (counted from the end).
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "resnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("ResNet50 base model layer not found within the model graph.")

    # Unfreeze the full backbone first
    base_model.trainable = True

    # Re-freeze all but the top `num_layers`, keeping BN always frozen
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers)
    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[ResNet50] Unfrozen {n_trainable}/{total_layers} backbone layers for fine-tuning.")
