"""
Stochastic Depth (Drop Path) regularization for DermaAI.

Randomly drops entire residual branches during training, which
acts as a regularizer and improves generalization in deep networks.
Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
"""

import tensorflow as tf
from keras import layers


class StochasticDepth(layers.Layer):
    """Drop the residual branch with probability `drop_rate` during training.

    Usage in a residual block::

        x = StochasticDepth(drop_rate=0.2)(shortcut, residual, training=training)

    Args:
        drop_rate: Probability of dropping the residual path (0 = no drop).
    """

    def __init__(self, drop_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}")
        self.drop_rate = drop_rate

    def call(self, shortcut, residual, training=False):
        if not training or self.drop_rate == 0.0:
            return shortcut + residual

        keep_prob = 1.0 - self.drop_rate
        batch_size = tf.shape(shortcut)[0]
        # Random Bernoulli mask per sample
        random_tensor = tf.random.uniform([batch_size, 1, 1, 1])
        binary_tensor = tf.cast(random_tensor < keep_prob, tf.float32)
        # Scale during training (inverted dropout style)
        output = shortcut + (residual * binary_tensor / keep_prob)
        return output

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"drop_rate": self.drop_rate})
        return cfg


class StochasticDepthCallback(tf.keras.callbacks.Callback):
    """Linearly scale drop rate from 0 to max_drop_rate over training.

    This follows the linear decay schedule from the original paper.
    """

    def __init__(self, stochastic_depth_layers: list, max_drop_rate: float = 0.5):
        super().__init__()
        self.sd_layers = stochastic_depth_layers
        self.max_drop_rate = max_drop_rate
        self._total_epochs = 1

    def on_train_begin(self, logs=None):
        self._total_epochs = self.params.get("epochs", 1) or 1

    def on_epoch_begin(self, epoch, logs=None):
        progress = epoch / max(self._total_epochs - 1, 1)
        current_rate = self.max_drop_rate * progress
        for layer in self.sd_layers:
            if hasattr(layer, "drop_rate"):
                layer.drop_rate = current_rate
