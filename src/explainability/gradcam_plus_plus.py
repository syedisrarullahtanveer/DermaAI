"""Grad-CAM++ (Generalized Gradient-based Visual Explanations).

Reference:
    Chattopadhyay, A., Sarkar, A., Howlader, P., & Balasubramanian, V. N. (2018).
    Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks.
    In WACV 2018.
"""

from typing import Optional
import numpy as np
import tensorflow as tf


class GradCAMPlusPlus:
    """Generate Grad-CAM++ heatmaps using higher-order gradients for precise lesion localization."""

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str) -> None:
        """Initialize Grad-CAM++ explainer.

        Args:
            model: Keras classification model.
            last_conv_layer_name: Name of target convolutional layer.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self._grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output],
        )

    def generate_heatmap(
        self, img_tensor: tf.Tensor, target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate normalized 2D Grad-CAM++ heatmap.

        Args:
            img_tensor: Input tensor of shape (1, H, W, 3).
            target_class_idx: Target class index (defaults to argmax prediction).

        Returns:
            np.ndarray: 2D heatmap array normalized to [0, 1].
        """
        with tf.GradientTape() as tape3:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape1:
                    conv_outputs, predictions = self._grad_model(img_tensor)
                    if target_class_idx is None:
                        target_class_idx = tf.argmax(predictions[0])
                    # Score for class c
                    score = predictions[:, target_class_idx]

                # 1st order gradients: dScore / dA^k
                first_grads = tape1.gradient(score, conv_outputs)
            # 2nd order gradients: d^2Score / (dA^k)^2
            second_grads = tape2.gradient(first_grads, conv_outputs)
        # 3rd order gradients: d^3Score / (dA^k)^3
        third_grads = tape3.gradient(second_grads, conv_outputs)

        if second_grads is None or third_grads is None:
            # Fall back to first-order Grad-CAM weighting if higher-order gradients vanish
            weights = tf.reduce_mean(first_grads, axis=(0, 1, 2))
            cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)
        else:
            # Alpha coefficients for Grad-CAM++
            # alpha^k_{i,j} = (d^2S / dA^2) / (2 * d^2S / dA^2 + sum(A^k) * d^3S / dA^3 + eps)
            eps = 1e-10
            sum_acts = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
            denom = 2.0 * second_grads + sum_acts * third_grads
            denom = tf.where(denom != 0.0, denom, eps)
            alphas = second_grads / denom

            # Positive gradients weighting
            pos_grads = tf.maximum(first_grads, 0.0)
            weights = tf.reduce_sum(alphas * pos_grads, axis=(1, 2))[0]
            cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)

        cam = tf.maximum(cam, 0.0)
        max_val = tf.math.reduce_max(cam)
        if max_val > 0.0:
            cam = cam / max_val

        return cam.numpy()
