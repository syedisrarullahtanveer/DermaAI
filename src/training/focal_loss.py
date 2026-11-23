"""Multi-Class Categorical Focal Loss for Severe Dermatological Class Imbalance.

Reference:
    Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollar, P. (2017).
    Focal Loss for Dense Object Detection. In ICCV 2017.
"""

from typing import Optional, Union, List
import tensorflow as tf
from tensorflow.keras import losses


@tf.keras.utils.register_keras_serializable(package="DermaAI")
class CategoricalFocalLoss(losses.Loss):
    """Categorical Focal Loss downweights well-classified examples to focus training on hard/rare lesions."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[Union[float, List[float]]] = None,
        from_logits: bool = False,
        name: str = "categorical_focal_loss",
        **kwargs,
    ):
        """Initialize Categorical Focal Loss.

        Args:
            gamma: Focusing parameter for modulating factor (1 - p_t)^gamma. gamma >= 0.
            alpha: Balancing factor or class-specific weights.
            from_logits: Whether y_pred is expected to be a logits tensor or probability distribution.
        """
        super().__init__(name=name, **kwargs)
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        self.gamma = float(gamma)
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute focal loss between ground truth one-hot targets and predicted probabilities."""
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip probabilities to prevent log(0) and numerical instability
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # Compute cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Compute modulating factor (1 - p_t)^gamma
        modulating_factor = tf.math.pow(1.0 - y_pred, self.gamma)

        focal_loss = modulating_factor * cross_entropy

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_weights = tf.constant(self.alpha, dtype=tf.float32)
                focal_loss = focal_loss * alpha_weights
            else:
                focal_loss = focal_loss * self.alpha

        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "from_logits": self.from_logits,
        })
        return config
