"""Integrated Gradients (Axiomatic Attribution for Deep Networks).

Reference:
    Sundararajan, M., Taly, A., & Yan, Q. (2017).
    Axiomatic Attribution for Deep Networks. In ICML 2017.
"""

from typing import Optional
import numpy as np
import tensorflow as tf


class IntegratedGradientsExplainer:
    """Generate axiomatic pixel attributions via straight-line path gradient integration."""

    def __init__(self, model: tf.keras.Model, num_steps: int = 20) -> None:
        """Initialize Integrated Gradients explainer.

        Args:
            model: Keras classification model.
            num_steps: Number of Riemann integral interpolation steps (default 20).
        """
        self.model = model
        self.num_steps = num_steps

    def generate_attributions(
        self,
        img_tensor: tf.Tensor,
        baseline: Optional[tf.Tensor] = None,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """Compute integrated gradient attributions for input image.

        Args:
            img_tensor: Input image tensor of shape (1, H, W, 3).
            baseline: Optional baseline image tensor (defaults to all zeros / black).
            target_class_idx: Target class index.

        Returns:
            np.ndarray: Pixel attribution map of shape (H, W, 3).
        """
        if baseline is None:
            baseline = tf.zeros_like(img_tensor)

        # Generate linearly interpolated path images: alphas from 0 to 1
        alphas = tf.linspace(0.0, 1.0, self.num_steps + 1)
        # Interpolated batch: shape (num_steps + 1, H, W, 3)
        interpolated_images = [
            baseline + alpha * (img_tensor - baseline) for alpha in alphas
        ]
        interpolated_batch = tf.concat(interpolated_images, axis=0)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_batch)
            predictions = self.model(interpolated_batch)
            if target_class_idx is None:
                target_class_idx = tf.argmax(predictions[0])
            target_scores = predictions[:, target_class_idx]

        grads = tape.gradient(target_scores, interpolated_batch)
        # Approximate integral using trapezoidal rule or Riemann average
        avg_grads = tf.reduce_mean(grads, axis=0, keepdims=True)

        # Multiply by input difference (x - x')
        attributions = (img_tensor - baseline) * avg_grads
        return attributions[0].numpy()
