"""
PolyLoss for fine-grained dermoscopic classification.
Reference: Leng et al., "PolyLoss: A Polynomial Expansion Perspective of Classification Loss Functions", ICLR 2022.
"""
import tensorflow as tf

class PolyLoss(tf.keras.losses.Loss):
    def __init__(self, epsilon: float = 1.0, name: str = "poly_loss"):
        super().__init__(name=name)
        self.epsilon = epsilon

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
        # Cross entropy component
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        # 1st polynomial expansion term: eps * (1 - pt)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        poly = ce + self.epsilon * (1.0 - pt)
        return tf.reduce_mean(poly)
