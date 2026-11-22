"""Transfer Learning Model Factory for Skin Lesion Classification.

Supports ResNet50, EfficientNetB0/B3, and MobileNetV2 with fine-tuning controls.
"""

from typing import Dict, List, Tuple
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB3, MobileNetV2


BACKBONE_REGISTRY = {
    "resnet50": ResNet50,
    "efficientnet_b0": EfficientNetB0,
    "efficientnet_b3": EfficientNetB3,
    "mobilenet_v2": MobileNetV2,
}


def get_available_backbones() -> List[str]:
    """Return list of supported pre-trained transfer learning backbones."""
    return list(BACKBONE_REGISTRY.keys())


def build_transfer_model(
    backbone_name: str = "efficientnet_b0",
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 7,
    weights: str = "imagenet",
    freeze_base: bool = True,
    dropout_rate: float = 0.3,
    dense_units: int = 256,
) -> tf.keras.Model:
    """Build a transfer learning architecture with pre-trained ImageNet feature extractor.

    Args:
        backbone_name: Model architecture ('resnet50', 'efficientnet_b0', 'mobilenet_v2').
        input_shape: Shape tuple of input dermoscopy images.
        num_classes: Number of target lesion classes.
        weights: Pre-trained weights ('imagenet' or None).
        freeze_base: Whether to freeze backbone weights during stage 1 feature extraction.
        dropout_rate: Dropout rate for classification neck.
        dense_units: Units in intermediate dense projection layer.

    Returns:
        tf.keras.Model: Configured transfer learning model.
    """
    key = backbone_name.lower().replace("-", "_")
    if key not in BACKBONE_REGISTRY:
        raise ValueError(f"Unknown backbone: '{backbone_name}'. Available: {get_available_backbones()}")

    backbone_cls = BACKBONE_REGISTRY[key]
    base_model = backbone_cls(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
    )
    base_model.trainable = not freeze_base

    inputs = layers.Input(shape=input_shape, name="image_input")
    x = base_model(inputs, training=not freeze_base)

    # Classification neck
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="neck_bn")(x)
    x = layers.Dense(dense_units, activation="relu", kernel_initializer="he_normal", name="neck_dense")(x)
    x = layers.Dropout(dropout_rate, name="neck_dropout")(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="classification_output",
    )(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"DermaAI_{key}")
    return model


def unfreeze_top_layers(model: tf.keras.Model, num_layers: int = 30) -> None:
    """Unfreeze the top N layers of the backbone for fine-tuning with differential learning rate."""
    # Find base model layer
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        # Fallback to model layers directly
        base_model = model

    base_model.trainable = True
    # Freeze all layers except top `num_layers`
    for layer in base_model.layers[:-num_layers]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            # Keep batch normalization running stats frozen to preserve stability
            layer.trainable = False
