"""Specialized Loss Functions for Extreme Dermatological Class Imbalance.

Implements Multi-Class Focal Loss and Weighted Categorical Cross-Entropy with Label Smoothing.
"""

from typing import Dict, List, Optional, Union
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses


class MultiClassFocalLoss(losses.Loss):
    """Multi-Class Focal Loss for addressing heavy class imbalance in dermoscopy datasets.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights easy examples (e.g. abundant 'nv' benign nevi) to focus gradients on hard
    malignant cases (e.g. rare 'mel' melanoma and 'bcc' carcinoma).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        name: str = "multi_class_focal_loss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent log(0) numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Cross entropy term: -log(p_t)
        ce = -tf.math.log(y_pred)

        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = tf.pow(1.0 - y_pred, self.gamma)

        # Compute focal loss
        focal_loss = modulating_factor * ce

        # Apply class weighting alpha if provided
        if self.alpha is not None:
            alpha_tensor = tf.constant(self.alpha, dtype=tf.float32)
            focal_loss = alpha_tensor * focal_loss

        # Multiply by true one-hot targets and average over batch
        loss = tf.reduce_sum(y_true * focal_loss, axis=-1)
        return tf.reduce_mean(loss)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


# Convenient alias
FocalLoss = MultiClassFocalLoss



def compute_balanced_class_weights(class_counts: Dict[str, int]) -> Dict[int, float]:
    """Compute inverse frequency class weights to balance gradient updates during training.

    weight_c = total_samples / (num_classes * count_c)
    """
    total = sum(class_counts.values())
    num_classes = len(class_counts)
    weights = {}
    for idx, (cls_name, count) in enumerate(class_counts.items()):
        weights[idx] = float(total / (num_classes * max(count, 1)))
    return weights


def get_loss_function(
    loss_name: str = "focal_loss",
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    label_smoothing: float = 0.05,
) -> losses.Loss:
    """Loss factory function."""
    key = loss_name.lower().replace("-", "_")
    if key == "focal_loss":
        return MultiClassFocalLoss(gamma=gamma, alpha=alpha)
    elif key in ["categorical_crossentropy", "ce"]:
        return losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    else:
        raise ValueError(f"Unsupported loss function: '{loss_name}'")
