"""Exponential Moving Average (EMA) Weight Averaging Callback for Model Generalization.

Reference:
    Tarvainen, A., & Valpola, H. (2017). Mean teachers are better role models:
    Weight-averaged consistency targets improve semi-supervised deep learning results.
    In NeurIPS 2017.
"""

from typing import List
import numpy as np
import tensorflow as tf


class ExponentialMovingAverageCallback(tf.keras.callbacks.Callback):
    """Maintains an exponential moving average of trainable model weights during training."""

    def __init__(self, decay: float = 0.999) -> None:
        """Initialize EMA callback.

        Args:
            decay: Smoothing rate for weight updates (default 0.999).
        """
        super().__init__()
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be between 0 and 1, got {decay}")
        self.decay = decay
        self.ema_weights: List[np.ndarray] = []

    def on_train_begin(self, logs=None):
        """Initialize shadow copy of model weights."""
        self.ema_weights = [w.copy() for w in self.model.get_weights()]

    def on_batch_end(self, batch, logs=None):
        """Update EMA weights after every training batch."""
        current_weights = self.model.get_weights()
        for i in range(len(self.ema_weights)):
            self.ema_weights[i] = (
                self.decay * self.ema_weights[i] + (1.0 - self.decay) * current_weights[i]
            )

    def apply_ema_weights(self):
        """Load shadow EMA weights into model for evaluation or inference."""
        if self.ema_weights:
            self.model.set_weights(self.ema_weights)
