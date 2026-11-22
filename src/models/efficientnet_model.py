"""EfficientNetB0 Transfer Learning Model for Skin Lesion Classification.

EfficientNetB0 provides a strong baseline in the EfficientNet family, balancing
accuracy and compute efficiency via compound scaling. Its built-in rescaling
layer handles pixel normalization, so no manual preprocessing is needed.
"""

from typing import List, Optional, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnet(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    weights: str = "imagenet",
    freeze_base: bool = True,
    dense_units: List[int] = None,
    dropout_rate: float = 0.3,
    name: str = "EfficientNetB0_Derma",
    freeze_backbone: Optional[bool] = None,
    **kwargs,
) -> tf.keras.Model:
    """Build EfficientNetB0-based transfer learning model for dermoscopy classification.

    EfficientNet applies its own internal rescaling (0–255 → 0–1 normalization),
    so raw uint8 OR float32 [0, 1] inputs are both accepted.

    Args:
        input_shape: Input image dimensions (H, W, C). Recommended: (224, 224, 3).
        num_classes: Number of target dermoscopy lesion classes.
        weights: 'imagenet' for pre-trained weights or None for training from scratch.
        freeze_base: Freeze backbone during stage-1 feature-extraction training.
        dense_units: Units in the classification head dense layers.
        dropout_rate: Spatial dropout applied before the final softmax layer.
        name: Keras model name string.

    Returns:
        tf.keras.Model: EfficientNetB0 model ready for compilation.
    """
    if freeze_backbone is not None:
        freeze_base = freeze_backbone
    if dense_units is None:
        dense_units = [256]

    # EfficientNetB0 backbone (no top)
    try:
        base_model = EfficientNetB0(
            include_top=False,
            weights=weights,
            input_shape=input_shape,
        )
    except Exception as exc:
        print(f"[EfficientNetB0] Could not load weights '{weights}' ({exc}). Initializing with weights=None.")
        base_model = EfficientNetB0(
            include_top=False,
            weights=None,
            input_shape=input_shape,
        )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=not freeze_base)

    # Feature aggregation with global average pooling
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


def unfreeze_efficientnet_top(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze the top-N layers of the EfficientNetB0 backbone for fine-tuning.

    BatchNormalization layers are kept frozen to prevent internal covariate
    shift when resuming training with a small learning rate.

    Args:
        model: Compiled EfficientNetB0 Keras model.
        num_layers: Number of top backbone layers to make trainable.
    """
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNetB0 base model layer not found.")

    base_model.trainable = True
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers)

    for idx, layer in enumerate(base_model.layers):
        if idx < freeze_until or isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    n_trainable = sum(1 for l in base_model.layers if l.trainable)
    print(f"[EfficientNetB0] Unfrozen {n_trainable}/{total_layers} backbone layers for fine-tuning.")


# Convenient alias matching standard naming conventions
build_efficientnet_b0 = build_efficientnet

