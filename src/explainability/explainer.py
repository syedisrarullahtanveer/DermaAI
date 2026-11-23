"""High‑level XAI orchestrator for DermaAI.

Provides a convenient API that:
  * Loads a trained Keras model (path to .keras or .h5 checkpoint).
  * Runs Grad‑CAM (or Grad‑CAM++) on a single image (and optional metadata).
  * Overlays the heatmap on the original image.
  * Packages the result (heatmap, overlay, class probabilities, disclaimer).
  * Saves the resulting visual artefacts to a user‑specified directory.

All visual artefacts are stored under the given ``output_dir`` with a
sub‑directory named after the source image filename.
"""

from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from src.explainability.gradcam import GradCAMExplainer
from src.explainability.visualizer import HeatmapVisualizer


@dataclass
class ExplanationResult:
    """Container for a single explanation.

    Attributes
    ----------
    image_path: str
        Path to the original input image.
    heatmap: np.ndarray
        Normalised 2‑D heatmap (H, W) in ``[0, 1]``.
    overlay: np.ndarray
        RGB image (uint8) with heatmap blended on top of the original.
    class_idx: int
        Predicted class index.
    class_probs: np.ndarray
        Probability distribution over all classes.
    disclaimer: str
        Mandatory disclaimer string – never omit.
    """

    image_path: str
    heatmap: np.ndarray
    overlay: np.ndarray
    class_idx: int
    class_probs: np.ndarray
    disclaimer: str = GradCAMExplainer.DISCLAIMER


class DermaExplainer:
    """Convenient wrapper around :class:`GradCAMExplainer`.

    Typical usage::

        expl = DermaExplainer(model_path="reports/checkpoints/best_resnet50.keras",
                              class_names=["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"])
        result = expl.explain(image_path="dataset/raw/sample_images/ISIC_SAMPLE_00001.jpg")
        expl.save_explanation(result, output_dir="reports/xai")
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[list[str]] = None,
        target_layer_name: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.class_names = class_names
        # Load model – we deliberately use ``compile=False`` because we only need inference.
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.gradcam = GradCAMExplainer(self.model, target_layer_name=target_layer_name)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def explain(
        self,
        image_path: str,
        metadata: Optional[np.ndarray] = None,
        use_gradcam_plus_plus: bool = False,
    ) -> ExplanationResult:
        """Generate an explanation for a single image.

        Parameters
        ----------
        image_path: str
            Path to an RGB image file readable by ``tf.keras.preprocessing.image.load_img``.
        metadata: Optional[np.ndarray]
            Optional 1‑D metadata vector. If ``None`` and the model expects a second input,
            a zero‑filled placeholder will be used automatically by ``GradCAMExplainer``.
        use_gradcam_plus_plus: bool
            Switch to the higher‑order Grad‑CAM++ algorithm.
        """
        # Load and preprocess the image – we reuse the project's ``utils`` if present or a simple decoder.
        img_raw = tf.keras.utils.load_img(image_path)
        img_arr = tf.keras.utils.img_to_array(img_raw)
        # Normalise to [0, 1] as the training pipeline expects.
        img_arr = img_arr / 255.0
        # Add batch dimension
        img_tensor = tf.expand_dims(img_arr, 0)

        heatmap = self.gradcam.generate_heatmap(
            img_tensor,
            class_idx=None,
            use_gradcam_plus_plus=use_gradcam_plus_plus,
            metadata_tensor=metadata,
        )
        overlay = HeatmapVisualizer.overlay_heatmap(img_arr, heatmap)
        class_idx, probs = self.gradcam.predict(img_tensor, metadata_tensor=metadata)
        return ExplanationResult(
            image_path=image_path,
            heatmap=heatmap,
            overlay=overlay,
            class_idx=int(class_idx),
            class_probs=probs,
        )

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_explanation(
        self, result: ExplanationResult, output_dir: str, prefix: Optional[str] = None
    ) -> None:
        """Write the heatmap, overlay and a small JSON manifest to ``output_dir``.

        The directory structure will be ``{output_dir}/{basename_without_ext}``.
        Files written:
            * ``original.png`` – copy of the input image (for completeness).
            * ``heatmap.npy`` – raw float array.
            * ``heatmap.png`` – visual heatmap coloured with ``jet``.
            * ``overlay.png`` – blended image.
            * ``manifest.json`` – JSON with predicted class, probabilities and disclaimer.
        """
        os.makedirs(output_dir, exist_ok=True)
        base = pathlib.Path(result.image_path).stem
        if prefix:
            base = f"{prefix}_{base}"
        out_path = pathlib.Path(output_dir) / base
        out_path.mkdir(parents=True, exist_ok=True)

        # Save original image (copy)
        orig = tf.keras.utils.load_img(result.image_path)
        orig_arr = tf.keras.utils.img_to_array(orig).astype(np.uint8)
        tf.keras.utils.save_img(str(out_path / "original.png"), orig_arr)

        # Save heatmap (raw + visual)
        np.save(str(out_path / "heatmap.npy"), result.heatmap)
        # Visual colour map – reuse visualizer's internal colormap logic via a temporary call
        heat_vis = HeatmapVisualizer.overlay_heatmap(
            np.zeros_like(result.overlay),  # black background just to get coloured heatmap
            result.heatmap,
            alpha=1.0,  # full colour, no blending
        )
        tf.keras.utils.save_img(str(out_path / "heatmap.png"), heat_vis)

        # Save overlay
        tf.keras.utils.save_img(str(out_path / "overlay.png"), result.overlay)

        # Manifest JSON
        manifest = {
            "image_path": result.image_path,
            "predicted_class_idx": result.class_idx,
            "predicted_class_name": (self.class_names[result.class_idx] if self.class_names else None),
            "class_probabilities": result.class_probs.tolist(),
            "disclaimer": result.disclaimer,
        }
        import json

        with open(out_path / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Batch helper (optional convenience)
    # ------------------------------------------------------------------
    def batch_explain(
        self,
        image_paths: list[str],
        output_dir: str,
        metadata_list: Optional[list[np.ndarray]] = None,
    ) -> None:
        """Run ``explain`` over a list of images and persist each result.

        ``metadata_list`` must be either ``None`` (no metadata) or a list matching
        ``image_paths`` length. If the model expects metadata but ``metadata_list``
        is ``None`` then zero‑filled placeholders are used automatically.
        """
        if metadata_list is None:
            metadata_list = [None] * len(image_paths)
        for img_path, meta in zip(image_paths, metadata_list):
            result = self.explain(img_path, metadata=meta)
            self.save_explanation(result, output_dir)

# End of file
