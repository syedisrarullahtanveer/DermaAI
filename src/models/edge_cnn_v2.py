import tensorflow as tf
from tensorflow.keras import layers, Model

def build_edge_cnn_v2(input_shape=(224,224,3), num_classes=7):
    """A lightweight edge‑optimized CNN.

    Architecture:
    - Conv2D 16 filters, 3×3, ReLU
    - Depthwise separable conv 32 filters
    - GlobalAveragePooling
    - Dense(num_classes) with softmax
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    x = layers.SeparableConv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs, name='edge_cnn_v2')
