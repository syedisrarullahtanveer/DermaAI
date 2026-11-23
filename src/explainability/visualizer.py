"""Heatmap Overlay and Prediction Visualization Module for Explainable AI.

Provides:
  - HeatmapVisualizer: Alpha-blends Grad-CAM heatmaps onto dermoscopic images.
  - plot_prediction_panel: Multi-panel figure (original | heatmap | overlay + bar chart).
  - save_explanation_artifacts: Saves all explanation outputs to a structured directory.

Design Principles:
  - All visual outputs include a disclaimer that model outputs are NOT diagnoses.
  - Functions are stateless (static/module-level) for easy unit testing.
  - OpenCV (cv2) is used for colormap + blending; matplotlib for panel figures.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# Disclaimer appended to all explanation figures
_DISCLAIMER = (
    "⚠  Research tool only. Model output is NOT a confirmed medical diagnosis.\n"
    "Consult a qualified dermatologist for clinical assessment."
)


class HeatmapVisualizer:
    """Superimposes Grad-CAM heatmaps onto dermoscopy images with configurable opacity and colormaps.

    Mathematical blending:
        overlay = α · colorize(heatmap) + (1 - α) · original_image
    where α is the ``alpha`` transparency parameter.
    """

    # Supported OpenCV colormaps for the heatmap
    COLORMAP_OPTIONS: Dict[str, int] = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "plasma": cv2.COLORMAP_PLASMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
    }

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: Union[str, int] = "jet",
    ) -> np.ndarray:
        """Blend a 2-D activation heatmap over an RGB dermoscopic image.

        Args:
            image: Original RGB image array [H, W, 3]. Values may be [0, 1] float
                or [0, 255] uint8.
            heatmap: 2-D normalized activation map [h, w] with values in [0, 1].
            alpha: Heatmap transparency factor (0.0 = invisible, 1.0 = full overlay).
            colormap: Colormap name string ('jet', 'hot', 'plasma', …) or an
                OpenCV COLORMAP_* constant integer.

        Returns:
            np.ndarray: RGB uint8 blended image of shape [H, W, 3].
        """
        # ---- Normalize image to uint8 -------------------------------------------
        if image.dtype in (np.float32, np.float64):
            orig_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            orig_uint8 = np.asarray(image, dtype=np.uint8).copy()

        h, w = orig_uint8.shape[:2]

        # ---- Resolve colormap -------------------------------------------------------
        if isinstance(colormap, str):
            cv_cmap = HeatmapVisualizer.COLORMAP_OPTIONS.get(
                colormap.lower(), cv2.COLORMAP_JET
            )
        else:
            cv_cmap = int(colormap)

        # ---- Resize heatmap to match image spatial resolution ----------------------
        heatmap_clipped = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
        cam_uint8 = np.uint8(255 * cv2.resize(heatmap_clipped, (w, h), interpolation=cv2.INTER_LINEAR))

        # ---- Apply colormap → BGR → RGB --------------------------------------------
        colored_bgr = cv2.applyColorMap(cam_uint8, cv_cmap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # ---- Alpha blend: overlay = α·color + (1-α)·orig --------------------------
        blended = cv2.addWeighted(orig_uint8, 1.0 - alpha, colored_rgb, alpha, 0)
        return blended  # uint8, RGB

    @staticmethod
    def extract_salient_contour(
        heatmap: np.ndarray,
        target_size: Tuple[int, int],
        threshold: float = 0.5,
    ) -> List:
        """Extract boundary contours of the most salient lesion regions.

        Args:
            heatmap: 2-D normalized activation map [h, w] in [0, 1].
            target_size: (width, height) to resize heatmap before thresholding.
            threshold: Activation threshold above which pixels are considered salient.

        Returns:
            List of contour arrays (from cv2.findContours).
        """
        resized = cv2.resize(heatmap.astype(np.float32), target_size, interpolation=cv2.INTER_LINEAR)
        binary = (resized >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours)

    @staticmethod
    def colorize_heatmap(heatmap: np.ndarray, colormap: str = "jet") -> np.ndarray:
        """Convert a 2-D float heatmap to an RGB uint8 image using a colormap.

        Args:
            heatmap: Normalized float array [h, w] in [0, 1].
            colormap: Colormap name string.

        Returns:
            np.ndarray: RGB uint8 array of shape [h, w, 3].
        """
        cv_cmap = HeatmapVisualizer.COLORMAP_OPTIONS.get(colormap.lower(), cv2.COLORMAP_JET)
        cam_uint8 = np.uint8(255 * np.clip(heatmap, 0.0, 1.0))
        bgr = cv2.applyColorMap(cam_uint8, cv_cmap)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Module-level visualization functions
# ---------------------------------------------------------------------------

def plot_prediction_panel(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_probs: np.ndarray,
    class_names: Sequence[str],
    predicted_class: str,
    confidence: float,
    save_path: Optional[Union[str, Path]] = None,
    image_id: str = "sample",
) -> plt.Figure:
    """Create a 4-panel prediction explanation figure.

    Layout::
        [Original Image] [Grad-CAM Heatmap] [Overlay] [Class Probability Bar Chart]

    Args:
        image: Original RGB image [H, W, 3] in [0, 255] uint8 or [0, 1] float.
        heatmap: 2-D float Grad-CAM map [h, w] in [0, 1].
        overlay: RGB uint8 blended image [H, W, 3].
        class_probs: Softmax probabilities array of length n_classes.
        class_names: Sequence of class name strings (length n_classes).
        predicted_class: Human-readable name of the top predicted class.
        confidence: Confidence score (0–1) of the top predicted class.
        save_path: Optional file path (.png/.pdf) to save the figure.
        image_id: Identifier used in the figure title.

    Returns:
        plt.Figure: The composed matplotlib figure object.
    """
    # Normalize image for display
    if image.dtype in (np.float32, np.float64):
        display_img = np.clip(image, 0.0, 1.0)
    else:
        display_img = image.astype(np.uint8)

    # Resize heatmap to image size for display
    h, w = display_img.shape[:2]
    heatmap_display = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, wspace=0.08)

    # ---- Panel 1: Original Image ------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(display_img)
    ax1.set_title("Original Image", fontsize=12, fontweight="bold")
    ax1.axis("off")

    # ---- Panel 2: Grad-CAM Heatmap ----------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(heatmap_display, cmap="jet", vmin=0, vmax=1)
    ax2.set_title("Grad-CAM Heatmap", fontsize=12, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label="Activation")

    # ---- Panel 3: Overlay -------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(overlay)
    ax3.set_title("Heatmap Overlay", fontsize=12, fontweight="bold")
    ax3.axis("off")

    # ---- Panel 4: Class Probability Bar Chart -----------------------------------
    ax4 = fig.add_subplot(gs[0, 3])
    n_classes = len(class_names)
    colors = ["#e74c3c" if cn == predicted_class else "#3498db" for cn in class_names]
    bars = ax4.barh(
        range(n_classes),
        class_probs[:n_classes],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )
    ax4.set_yticks(range(n_classes))
    ax4.set_yticklabels(class_names, fontsize=9)
    ax4.set_xlim(0, 1.05)
    ax4.set_xlabel("Probability", fontsize=10)
    ax4.set_title("Class Probabilities", fontsize=12, fontweight="bold")
    ax4.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # Annotate bar values
    for bar, prob in zip(bars, class_probs[:n_classes]):
        ax4.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2.0,
            f"{prob:.3f}",
            va="center",
            fontsize=8,
        )

    # Legend
    pred_patch = mpatches.Patch(color="#e74c3c", label=f"Predicted: {predicted_class}")
    other_patch = mpatches.Patch(color="#3498db", label="Other classes")
    ax4.legend(handles=[pred_patch, other_patch], fontsize=8, loc="lower right")

    # ---- Figure Title & Disclaimer ----------------------------------------------
    fig.suptitle(
        f"DermaAI Explanation  |  ID: {image_id}  |  "
        f"Predicted: {predicted_class}  ({confidence * 100:.1f}% confidence)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )
    fig.text(
        0.5, -0.04,
        _DISCLAIMER,
        ha="center",
        fontsize=8,
        color="#c0392b",
        style="italic",
        wrap=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Prediction panel saved → %s", save_path)

    return fig


def save_explanation_artifacts(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_probs: np.ndarray,
    class_names: Sequence[str],
    predicted_class: str,
    confidence: float,
    output_dir: Union[str, Path],
    image_id: str = "sample",
    extra_meta: Optional[Dict] = None,
) -> Dict[str, Path]:
    """Save all explanation artifacts to a structured directory.

    Saved files::
        <output_dir>/<image_id>/
            original.png
            heatmap.png
            overlay.png
            panel.png
            metadata.json

    Args:
        image: Original RGB image [H, W, 3].
        heatmap: 2-D float Grad-CAM map [h, w] in [0, 1].
        overlay: RGB uint8 blended image [H, W, 3].
        class_probs: Softmax probabilities array.
        class_names: Sequence of class name strings.
        predicted_class: Top predicted class name.
        confidence: Confidence score of top class.
        output_dir: Root directory for saving artifacts.
        image_id: Unique identifier for this explanation.
        extra_meta: Optional extra metadata dict to include in metadata.json.

    Returns:
        Dict mapping artifact name → Path of saved file.
    """
    import json

    out = Path(output_dir) / image_id
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Path] = {}

    # -- Original image ----------------------------------------------------------
    orig_path = out / "original.png"
    if image.dtype in (np.float32, np.float64):
        orig_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    else:
        orig_uint8 = image.astype(np.uint8)
    cv2.imwrite(str(orig_path), cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR))
    saved["original"] = orig_path

    # -- Heatmap -----------------------------------------------------------------
    heatmap_path = out / "heatmap.png"
    h, w = orig_uint8.shape[:2]
    heatmap_resized = cv2.resize(heatmap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_colored = HeatmapVisualizer.colorize_heatmap(heatmap_resized, colormap="jet")
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(heatmap_colored, cv2.COLOR_RGB2BGR))
    saved["heatmap"] = heatmap_path

    # -- Overlay -----------------------------------------------------------------
    overlay_path = out / "overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    saved["overlay"] = overlay_path

    # -- Prediction panel --------------------------------------------------------
    panel_path = out / "panel.png"
    fig = plot_prediction_panel(
        image=image,
        heatmap=heatmap,
        overlay=overlay,
        class_probs=class_probs,
        class_names=class_names,
        predicted_class=predicted_class,
        confidence=confidence,
        save_path=panel_path,
        image_id=image_id,
    )
    plt.close(fig)
    saved["panel"] = panel_path

    # -- Metadata JSON -----------------------------------------------------------
    meta_path = out / "metadata.json"
    meta = {
        "image_id": image_id,
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "class_probabilities": {
            name: float(prob)
            for name, prob in zip(class_names, class_probs[: len(class_names)])
        },
        "disclaimer": _DISCLAIMER,
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    saved["metadata"] = meta_path

    logger.info("Explanation artifacts saved to %s", out)
    return saved
