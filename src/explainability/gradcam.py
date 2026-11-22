"""Grad-CAM and Grad-CAM++ Implementations for Dermatological Lesion Localization."""

from typing import Optional, Tuple
import numpy as np
import tensorflow as tf


class GradCAMExplainer:
    """Generates Gradient-weighted Class Activation Maps (Grad-CAM & Grad-CAM++).

    Visualizes which visual regions within the dermoscopic field of view
    contribute most significantly to the network's diagnostic decision.
    """

    def __init__(self, model: tf.keras.Model, target_layer_name: Optional[str] = None):
        self.model = model
        self.target_layer_name = target_layer_name or self._find_target_conv_layer()
        self.grad_model = self._create_grad_model()

    def _find_target_conv_layer(self) -> str:
        """Automatically identify the final 4D convolutional feature layer in the model."""
        for layer in reversed(self.model.layers):
            # Check if layer outputs a 4D feature map (Batch, Height, Width, Channels)
            if hasattr(layer, "output_shape") and len(layer.output_shape) == 4:
                return layer.name
            # If nested model (e.g. Backbone)
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if hasattr(sub_layer, "output_shape") and len(sub_layer.output_shape) == 4:
                        return sub_layer.name
        raise ValueError("Could not find a valid 4D convolutional layer for Grad-CAM.")

    def _create_grad_model(self) -> tf.keras.Model:
        """Create a gradient model mapping inputs to both target feature maps and class scores."""
        target_layer = self.model.get_layer(self.target_layer_name)
        return tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output],
        )

    def generate_heatmap(
        self,
        image_tensor: tf.Tensor,
        class_idx: Optional[int] = None,
        use_gradcam_plus_plus: bool = False,
    ) -> np.ndarray:
        """Compute Grad-CAM activation heatmap for a given input image and target class.

        Args:
            image_tensor: Preprocessed image tensor with shape [1, H, W, 3].
            class_idx: Target class index. If None, uses top predicted class.
            use_gradcam_plus_plus: Whether to use higher-order derivatives (Grad-CAM++).

        Returns:
            np.ndarray: Normalized 2D activation heatmap in range [0, 1].
        """
        with tf.GradientTape(persistent=True) as tape:
            conv_outputs, predictions = self.grad_model(image_tensor)
            if class_idx is None:
                class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        # First-order gradients of class score with respect to feature maps
        grads = tape.gradient(loss, conv_outputs)

        if use_gradcam_plus_plus:
            # Grad-CAM++: weighting using second and third order gradients
            grads_2 = tape.gradient(grads, conv_outputs)
            grads_3 = tape.gradient(grads_2, conv_outputs)

            global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
            alpha_num = grads_2
            alpha_denom = 2.0 * grads_2 + grads_3 * global_sum + 1e-7
            alphas = alpha_num / alpha_denom

            weights = tf.reduce_sum(alphas * tf.maximum(grads, 0), axis=(1, 2), keepdims=True)
        else:
            # Standard Grad-CAM: Global average pooling over gradients
            weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)

        del tape

        # Weighted combination of forward activation maps
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]

        # ReLU: retain only features that positively influence the target class
        cam = tf.maximum(cam, 0)

        # Normalize to [0, 1]
        cam_max = tf.reduce_max(cam)
        if cam_max > 0:
            cam = cam / cam_max

        return cam.numpy()
