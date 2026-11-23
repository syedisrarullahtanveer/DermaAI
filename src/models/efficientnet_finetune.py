"""
EfficientNet fine-tuning wrapper for DermaAI.

Provides progressive unfreezing of EfficientNet layers for
transfer learning on dermoscopy datasets.
"""

import tensorflow as tf
from keras import layers, Model


def build_efficientnet_finetune(
    num_classes: int = 7,
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.3,
    variant: str = "B0",
    freeze_base: bool = True,
) -> Model:
    """Build an EfficientNet model with a custom classification head.

    Args:
        num_classes: Number of output skin disease classes.
        input_shape: Input image dimensions (H, W, C).
        dropout_rate: Dropout applied before the final Dense layer.
        variant: EfficientNet variant: 'B0' through 'B7'.
        freeze_base: If True, freeze all base model weights initially.

    Returns:
        Compiled Keras Model ready for fine-tuning.
    """
    variant_map = {
        "B0": tf.keras.applications.EfficientNetB0,
        "B1": tf.keras.applications.EfficientNetB1,
        "B2": tf.keras.applications.EfficientNetB2,
        "B3": tf.keras.applications.EfficientNetB3,
        "B4": tf.keras.applications.EfficientNetB4,
    }
    if variant not in variant_map:
        raise ValueError(f"Unsupported variant '{variant}'. Choose from {list(variant_map)}")

    base = variant_map[variant](
        include_top=False,
        weights=None,  # avoid downloading weights in offline environments
        input_shape=input_shape,
    )
    base.trainable = not freeze_base

    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs, name=f"DermaEfficientNet{variant}")


def unfreeze_top_layers(model: Model, num_layers: int = 20) -> Model:
    """Unfreeze the top N layers of the base model for fine-tuning.

    Args:
        model: Keras model with a frozen base sub-model.
        num_layers: Number of layers from the end to unfreeze.

    Returns:
        Model with the requested layers set trainable.
    """
    # Locate the EfficientNet base layer
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower():
            for sub in layer.layers[-num_layers:]:
                if not isinstance(sub, layers.BatchNormalization):
                    sub.trainable = True
            break
    return model
