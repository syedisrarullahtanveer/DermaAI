"""Grad-CAM and Grad-CAM++ Implementations for Dermatological Lesion Localization.

Mathematical Formulation
------------------------
Grad-CAM (Selvaraju et al., 2017):

1. Forward pass to obtain class-discriminative feature maps F ∈ R^{H × W × K}
   from the target convolutional layer.

2. Compute gradient of class score S^c (before softmax) w.r.t. each channel k:
       ∂S^c / ∂F_k

3. Global Average Pool the gradients to obtain importance weights:
       α_k^c = (1 / (H·W)) ∑_{i,j} (∂S^c / ∂F_k_{ij})

4. Weighted combination + ReLU (retain only positive contributions):
       CAM^c = ReLU( ∑_k α_k^c · F_k )

5. Normalize to [0, 1]:
       heatmap = (CAM^c - min) / (max - min)

Grad-CAM++ (Chattopadhay et al., 2018):
       Uses higher-order gradients (2nd, 3rd) for more precise pixel-level weights.

Multimodal Handling:
       When the model has multiple inputs (image + metadata), only the image
       branch is used for Grad-CAM. Metadata is passed through as dummy zeros
       if not supplied explicitly.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class GradCAMExplainer:
    """Generates Gradient-weighted Class Activation Maps (Grad-CAM & Grad-CAM++).

    Visualizes which visual regions within the dermoscopic field of view
    contribute most significantly to the network's classification decision.

    Usage:
        explainer = GradCAMExplainer(model)
        heatmap = explainer.generate_heatmap(image_tensor)  # shape: (H, W) in [0,1]

    Note:
        Heatmaps indicate regions the model attended to — they are NOT
        clinically validated indicators of pathology.
    """

    # Disclaimer embedded in every result — model outputs are NOT diagnoses
    DISCLAIMER: str = (
        "⚠ Research Tool Only: This Grad-CAM visualization highlights regions the AI model "
        "focused on. It is NOT a confirmed medical diagnosis. Always consult a qualified "
        "dermatologist for clinical assessment."
    )

    def __init__(
        self,
        model: tf.keras.Model,
        target_layer_name: Optional[str] = None,
    ) -> None:
        """Initialise Grad-CAM explainer.

        Args:
            model: Trained Keras classification model.
            target_layer_name: Name of target convolutional layer. If None,
                the last 4-D layer is auto-detected.
        """
        self.model = model
        self.target_layer_name: str = target_layer_name or self._find_target_conv_layer()
        logger.info("GradCAM target layer: %s", self.target_layer_name)
        self.grad_model: tf.keras.Model = self._create_grad_model()

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------

    def _find_target_conv_layer(self) -> str:
        """Auto-detect the last 4-D convolutional feature layer in the model.

        Searches the top-level model layers first, then recurses into any
        nested sub-models (e.g. backbone Model layers) to find the final
        spatial feature map producer.

        Returns:
            str: Name of the target layer.

        Raises:
            ValueError: If no 4-D output layer is found.
        """
        # Walk from the end – we want the *last* convolutional layer
        for layer in reversed(self.model.layers):
            if self._is_conv_layer(layer):
                return layer.name
            # Recurse into nested sub-models (backbone wrappers)
            if isinstance(layer, tf.keras.Model):
                for sub_layer in reversed(layer.layers):
                    if self._is_conv_layer(sub_layer):
                        return sub_layer.name

        raise ValueError(
            "GradCAMExplainer: Could not find a valid 4-D convolutional layer "
            f"in model '{self.model.name}'. Provide target_layer_name explicitly."
        )

    @staticmethod
    def _is_conv_layer(layer: tf.keras.layers.Layer) -> bool:
        """Return True if layer produces a 4-D spatial feature map."""
        try:
            shape = layer.output_shape
            # Accepts both single shape tuple and list of shapes (multi-output)
            if isinstance(shape, list):
                shape = shape[0]
            return len(shape) == 4
        except (AttributeError, RuntimeError):
            return False

    # ------------------------------------------------------------------
    # Gradient model construction
    # ------------------------------------------------------------------

    def _create_grad_model(self) -> tf.keras.Model:
        """Build auxiliary model that outputs (feature maps, class logits).

        The auxiliary model shares weights with self.model — no extra
        parameters are introduced.

        Returns:
            tf.keras.Model: Outputs [feature_map_tensor, predictions_tensor].

        Raises:
            ValueError: If target_layer_name is not found in the model graph.
        """
        # Try to get layer directly from top-level model
        target_layer = self._get_layer_by_name(self.model, self.target_layer_name)
        if target_layer is None:
            raise ValueError(
                f"Layer '{self.target_layer_name}' not found in model '{self.model.name}'. "
                "Available layers: "
                + ", ".join(l.name for l in self.model.layers)
            )

        return tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[target_layer.output, self.model.output],
        )

    @staticmethod
    def _get_layer_by_name(
        model: tf.keras.Model, name: str
    ) -> Optional[tf.keras.layers.Layer]:
        """Recursively search for a layer by name in a potentially nested model."""
        for layer in model.layers:
            if layer.name == name:
                return layer
            if isinstance(layer, tf.keras.Model):
                found = GradCAMExplainer._get_layer_by_name(layer, name)
                if found is not None:
                    return found
        return None

    # ------------------------------------------------------------------
    # Heatmap generation
    # ------------------------------------------------------------------

    def generate_heatmap(
        self,
        image_tensor: Union[tf.Tensor, np.ndarray],
        class_idx: Optional[int] = None,
        use_gradcam_plus_plus: bool = False,
        metadata_tensor: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> np.ndarray:
        """Compute a Grad-CAM or Grad-CAM++ activation heatmap.

        Mathematical summary:
            Standard Grad-CAM:
                α_k = GAP( ∂S^c / ∂F_k )
                CAM = ReLU( ∑_k α_k · F_k )

            Grad-CAM++ (2nd/3rd order):
                α_k^{++} = ∑_{ij} [ (∂²S^c/∂F_{k,ij}²) /
                             (2·∂²S^c/∂F_{k,ij}² + F_k · ∂³S^c/∂F_{k,ij}³ + ε) ]
                             · ReLU(∂S^c/∂F_{k,ij})

        Args:
            image_tensor: Preprocessed image tensor of shape [1, H, W, 3].
            class_idx: Target class index. Uses argmax(predictions) if None.
            use_gradcam_plus_plus: Use higher-order gradients for finer maps.
            metadata_tensor: Optional metadata array for multimodal models.
                Shape [1, meta_dim]. If the model expects metadata but this
                is None, zeros are supplied automatically.

        Returns:
            np.ndarray: 2-D normalized heatmap of shape (h, w) with values
            in [0, 1], where h, w are the spatial dims of the target layer.
        """
        # Ensure batch dimension
        if isinstance(image_tensor, np.ndarray):
            image_tensor = tf.constant(image_tensor, dtype=tf.float32)
        if image_tensor.ndim == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)

        # Build inputs list (handle multimodal models with >1 input)
        model_inputs = self._build_model_inputs(image_tensor, metadata_tensor)

        with tf.GradientTape(persistent=use_gradcam_plus_plus) as tape:
            # Watch feature maps explicitly (not Variables → need watch)
            tape.watch(image_tensor)

            conv_outputs, predictions = self.grad_model(model_inputs, training=False)
            tape.watch(conv_outputs)

            if class_idx is None:
                class_idx = int(tf.argmax(predictions[0]).numpy())

            # Class score (pre-softmax — use raw logit if available, else softmax prob)
            loss = predictions[:, class_idx]

        # ---- First-order gradients ∂loss/∂conv_outputs ----
        grads = tape.gradient(loss, conv_outputs)

        if use_gradcam_plus_plus:
            grads_2 = tape.gradient(grads, conv_outputs)
            grads_3 = tape.gradient(grads_2, conv_outputs)
            del tape

            # Denominator: 2·g² + F·g³
            global_sum = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
            alpha_num = grads_2
            alpha_denom = 2.0 * grads_2 + grads_3 * global_sum + 1e-7
            alphas = alpha_num / alpha_denom

            # Weight per-pixel gradients
            weights = tf.reduce_sum(
                alphas * tf.maximum(grads, 0.0), axis=(1, 2), keepdims=True
            )
        else:
            del tape
            # Standard Grad-CAM: α_k = GAP(grads)
            weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)

        # Weighted combination: CAM = ReLU(∑_k α_k · F_k)
        cam = tf.reduce_sum(weights * conv_outputs, axis=-1)[0]  # (h, w)
        cam = tf.maximum(cam, 0.0)  # ReLU

        # Normalize to [0, 1]
        cam_min = tf.reduce_min(cam)
        cam_max = tf.reduce_max(cam)
        denom = cam_max - cam_min
        if denom > 1e-8:
            cam = (cam - cam_min) / denom
        else:
            cam = tf.zeros_like(cam)  # Uniform feature map — all zeros

        return cam.numpy().astype(np.float32)

    def _build_model_inputs(
        self,
        image_tensor: tf.Tensor,
        metadata_tensor: Optional[Union[tf.Tensor, np.ndarray]],
    ) -> Union[tf.Tensor, List[tf.Tensor]]:
        """Construct the correct input list for single-input or multimodal models.

        Args:
            image_tensor: Image tensor [1, H, W, 3].
            metadata_tensor: Optional metadata tensor [1, meta_dim].

        Returns:
            Single tensor (vision-only) or list of tensors (multimodal).
        """
        n_inputs = len(self.model.inputs)

        if n_inputs == 1:
            return image_tensor

        # Multimodal: model expects [image, metadata]
        if metadata_tensor is None:
            # Infer metadata dim from model's second input shape
            meta_dim = self.model.inputs[1].shape[-1] or 10
            metadata_tensor = tf.zeros([1, meta_dim], dtype=tf.float32)
            logger.warning(
                "GradCAM: Multimodal model detected but no metadata_tensor provided. "
                "Using zero-filled metadata of shape [1, %d].", meta_dim
            )
        elif isinstance(metadata_tensor, np.ndarray):
            metadata_tensor = tf.constant(metadata_tensor, dtype=tf.float32)

        return [image_tensor, metadata_tensor]

    # ------------------------------------------------------------------
    # Convenience: get predicted class + probabilities
    # ------------------------------------------------------------------

    def predict(
        self,
        image_tensor: Union[tf.Tensor, np.ndarray],
        metadata_tensor: Optional[Union[tf.Tensor, np.ndarray]] = None,
    ) -> Tuple[int, np.ndarray]:
        """Run forward pass and return (class_idx, class_probabilities).

        Args:
            image_tensor: Preprocessed image [1, H, W, 3] or [H, W, 3].
            metadata_tensor: Optional metadata [1, meta_dim].

        Returns:
            Tuple of (predicted_class_index, probabilities_array).
        """
        if isinstance(image_tensor, np.ndarray):
            image_tensor = tf.constant(image_tensor, dtype=tf.float32)
        if image_tensor.ndim == 3:
            image_tensor = tf.expand_dims(image_tensor, 0)

        model_inputs = self._build_model_inputs(image_tensor, metadata_tensor)
        preds = self.model(model_inputs, training=False)
        probs = preds.numpy()[0]
        return int(np.argmax(probs)), probs

    # ------------------------------------------------------------------
    # Available layer names (utility)
    # ------------------------------------------------------------------

    def list_conv_layers(self) -> List[str]:
        """Return names of all 4-D convolutional layers in the model."""
        names: List[str] = []
        for layer in self.model.layers:
            if self._is_conv_layer(layer):
                names.append(layer.name)
            if isinstance(layer, tf.keras.Model):
                for sub in layer.layers:
                    if self._is_conv_layer(sub):
                        names.append(sub.name)
        return names
