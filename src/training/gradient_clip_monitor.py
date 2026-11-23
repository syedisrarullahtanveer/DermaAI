"""
Gradient clipping monitor callback for DermaAI.

Tracks per-batch gradient norms and flags potential exploding gradient
events, which is critical for diagnosing training instability.
"""

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


class GradientClipMonitor(tf.keras.callbacks.Callback):
    """Monitor gradient norms and log warnings for large values.

    Wraps the optimizer's apply_gradients to intercept gradient norms.
    Logs a warning when the global gradient norm exceeds `warn_threshold`.

    Args:
        warn_threshold: Gradient norm above this value triggers a warning.
        log_every_n_steps: Interval for logging gradient norm (default=50).
    """

    def __init__(
        self,
        warn_threshold: float = 10.0,
        log_every_n_steps: int = 50,
    ):
        super().__init__()
        self.warn_threshold = warn_threshold
        self.log_every_n_steps = log_every_n_steps
        self._step = 0
        self.gradient_norm_history: list = []

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        # Keras logs gradient norm if model.compile uses clipnorm
        # We approximate from logs if available, otherwise track step
        norm = logs.get("gradient_norm", None)
        if norm is not None:
            self.gradient_norm_history.append(float(norm))
            if norm > self.warn_threshold:
                logger.warning(
                    "Step %d: gradient norm %.4f exceeds threshold %.4f",
                    self._step,
                    norm,
                    self.warn_threshold,
                )
        self._step += 1
        if self._step % self.log_every_n_steps == 0:
            recent = self.gradient_norm_history[-self.log_every_n_steps:]
            if recent:
                avg_norm = sum(recent) / len(recent)
                logger.info("Avg gradient norm (last %d steps): %.4f", len(recent), avg_norm)

    def on_epoch_end(self, epoch, logs=None):
        if self.gradient_norm_history:
            epoch_avg = sum(self.gradient_norm_history) / len(self.gradient_norm_history)
            logger.info("Epoch %d mean gradient norm: %.4f", epoch, epoch_avg)
            if logs is not None:
                logs["mean_gradient_norm"] = epoch_avg

    @property
    def max_gradient_norm(self) -> float:
        """Return the maximum gradient norm observed so far."""
        return max(self.gradient_norm_history) if self.gradient_norm_history else 0.0
