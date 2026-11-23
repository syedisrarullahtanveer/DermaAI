"""
Multi-head self-attention token bridge: connects CNN feature maps to vision transformer tokens.
"""
import tensorflow as tf
from keras import layers

class MultiHeadTokenBridge(layers.Layer):
    def __init__(self, embed_dim: int = 64, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.project = layers.Dense(embed_dim)
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.layernorm = layers.LayerNormalization()

    def call(self, feature_map):
        # feature_map: (B, H, W, C)
        b, h, w, c = feature_map.shape[0], feature_map.shape[1], feature_map.shape[2], feature_map.shape[3]
        tokens = tf.reshape(feature_map, [-1, h * w, c])
        tokens = self.project(tokens)
        attended = self.mha(tokens, tokens)
        normed = self.layernorm(tokens + attended)
        return normed
