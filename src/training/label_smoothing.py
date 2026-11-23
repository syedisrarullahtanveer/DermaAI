"""
Label smoothing cross-entropy loss for DermaAI.

Prevents overconfident predictions by softening the target distribution.
Reference: Müller et al., "When Does Label Smoothing Help?" NeurIPS 2019.
"""

import tensorflow as tf


class LabelSmoothingCrossEntropy(tf.keras.losses.Loss):
    """Categorical cross-entropy with label smoothing.

    Args:
        smoothing: Float in [0, 1). Amount of label mass to distribute
            uniformly across all classes. 0.0 = standard cross-entropy.
        from_logits: Whether predictions are raw logits or probabilities.
        name: Name for this loss.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        from_logits: bool = False,
        name: str = "label_smoothing_cross_entropy",
    ):
        super().__init__(name=name)
        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0, 1), got {smoothing}")
        self.smoothing = smoothing
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
        # Smooth: (1 - eps) * one_hot + eps / K
        y_smooth = y_true * (1.0 - self.smoothing) + self.smoothing / num_classes

        if self.from_logits:
            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=y_smooth, logits=y_pred
            )
        else:
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
            loss = -tf.reduce_sum(y_smooth * tf.math.log(y_pred), axis=-1)

        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"smoothing": self.smoothing, "from_logits": self.from_logits})
        return cfg
