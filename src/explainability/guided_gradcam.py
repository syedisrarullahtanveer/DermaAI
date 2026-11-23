"""Guided Grad-CAM High-Resolution Saliency Map Generator.

Combines standard coarse Grad-CAM heatmaps with Guided Backpropagation gradients
to produce high-resolution, class-discriminative visualizations of dermatological lesions.

Reference:
    Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).
    Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.
    In ICCV 2017.
"""

from typing import Tuple, Optional
import cv2
import numpy as np
import tensorflow as tf


class GuidedGradCAMGenerator:
    """Generate fine-grained, class-discriminative Guided Grad-CAM visualizations."""

    def __init__(self, model: tf.keras.Model, last_conv_layer_name: str) -> None:
        """Initialize Guided Grad-CAM generator.

        Args:
            model: Keras classification model.
            last_conv_layer_name: Target convolutional layer name for Grad-CAM.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self._grad_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output],
        )

    def compute_gradcam_heatmap(
        self, img_array: tf.Tensor, target_class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Compute standard normalized 2D Grad-CAM heatmap."""
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self._grad_model(img_array)
            if target_class_idx is None:
                target_class_idx = tf.argmax(predictions[0])
            loss = predictions[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight feature maps by pooled gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        # Apply ReLU to keep only positive contributions
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val

        return heatmap.numpy()

    def generate(
        self,
        img_array: tf.Tensor,
        target_class_idx: Optional[int] = None,
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Generate high-resolution Guided Grad-CAM visualization.

        Args:
            img_array: Input image tensor of shape (1, H, W, 3).
            target_class_idx: Optional target class index.
            target_size: Output spatial dimensions (height, width).

        Returns:
            np.ndarray: Normalized RGB Guided Grad-CAM image with values in [0.0, 1.0].
        """
        # 1. Compute standard Grad-CAM heatmap
        heatmap = self.compute_gradcam_heatmap(img_array, target_class_idx)

        # 2. Resize heatmap to input target dimensions
        resized_heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]))

        # 3. Compute input gradients
        with tf.GradientTape() as tape:
            tape.watch(img_array)
            preds = self.model(img_array)
            if target_class_idx is None:
                target_class_idx = tf.argmax(preds[0])
            loss = preds[:, target_class_idx]

        raw_grads = tape.gradient(loss, img_array)[0].numpy()

        # 4. Rectify gradients (keep only positive gradients)
        rectified_grads = np.maximum(raw_grads, 0.0)

        # 5. Element-wise fusion of rectified gradients with 3-channel broadcasted heatmap
        heatmap_3ch = np.repeat(resized_heatmap[:, :, np.newaxis], 3, axis=-1)
        guided_gradcam = rectified_grads * heatmap_3ch

        # 6. Normalize to [0, 1]
        denom = np.max(guided_gradcam) - np.min(guided_gradcam)
        if denom > 1e-8:
            guided_gradcam = (guided_gradcam - np.min(guided_gradcam)) / denom

        return guided_gradcam.astype(np.float32)
