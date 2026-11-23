"""
ResNet50 transfer learning architecture with dual pooling (Avg + Max) head for dermoscopy.
"""
import tensorflow as tf
from keras import layers, Model

def build_resnet50_finetune(
    num_classes: int = 7,
    input_shape: tuple = (224, 224, 3),
    dropout_rate: float = 0.4,
    freeze_base: bool = True
) -> Model:
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape
    )
    base.trainable = not freeze_base
    
    inputs = tf.keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    
    # Dual pooling: captures both prominent focal features and background textures
    avg_pool = layers.GlobalAveragePooling2D()(x)
    max_pool = layers.GlobalMaxPooling2D()(x)
    combined = layers.Concatenate()([avg_pool, max_pool])
    
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(dropout_rate)(combined)
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.BatchNormalization()(combined)
    combined = layers.Dropout(dropout_rate / 2)(combined)
    outputs = layers.Dense(num_classes, activation="softmax")(combined)
    
    return Model(inputs, outputs, name="DermaResNet50DualPool")
