"""Visualization utilities for DermaAI XAI.

Provides a single helper ``plot_prediction_panel`` that creates a three‑panel
figure:

* Original image
* Grad‑CAM heatmap (colored)
* Overlay (image + heatmap)

and a bar‑chart of class probabilities. The figure is returned as a Matplotlib
``Figure`` object and can optionally be saved to disk.
"""

from __future__ import annotations

import os
import pathlib
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _apply_colormap(heatmap: np.ndarray, cmap: str = "jet") -> np.ndarray:
    """Convert a normalized (H, W) heatmap to an RGB image using Matplotlib's colormap.

    Args:
        heatmap: 2‑D array with values in ``[0, 1]``.
        cmap: Name of Matplotlib colormap (default ``jet``).

    Returns:
        RGB ``uint8`` array of shape ``(H, W, 3)``.
    """
    import matplotlib.cm as cm

    cmap_obj = cm.get_cmap(cmap)
    colored = cmap_obj(heatmap)[:, :, :3]  # drop alpha channel
    return (colored * 255).astype(np.uint8)


def plot_prediction_panel(
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    class_probs: np.ndarray,
    class_names: Optional[List[str]] = None,
    predicted_class: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a publication‑ready figure showing model prediction and explanation.

    Parameters
    ----------
    image: np.ndarray
        Original RGB image, ``[H, W, 3]``, values in ``[0, 255]`` or ``[0, 1]``.
    heatmap: np.ndarray
        Normalised heatmap ``[H, W]`` in ``[0, 1]``.
    overlay: np.ndarray
        Blended RGB image (same shape as ``image``) produced by
        ``HeatmapVisualizer.overlay_heatmap``.
    class_probs: np.ndarray
        1‑D array of class probabilities that sum to 1.
    class_names: list[str] | None
        Human‑readable class labels. If ``None`` indices are shown.
    predicted_class: int | None
        Index of the predicted class. If ``None`` the argmax of ``class_probs`` is used.
    save_path: str | None
        If provided, the figure is saved to this location (PNG format).

    Returns
    -------
    matplotlib.figure.Figure
        The generated Matplotlib figure.
    """
    if predicted_class is None:
        predicted_class = int(np.argmax(class_probs))

    # Ensure proper dtype/range for display
    def _norm_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0, 0, 255)
        return arr.astype(np.uint8)

    img_disp = _norm_uint8(image)
    overlay_disp = _norm_uint8(overlay)
    heat_rgb = _apply_colormap(heatmap)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img_disp)
    ax_img.axis("off")
    ax_img.set_title("Original")

    ax_heat = fig.add_subplot(gs[0, 1])
    ax_heat.imshow(heat_rgb)
    ax_heat.axis("off")
    ax_heat.set_title("Grad‑CAM heatmap")

    ax_overlay = fig.add_subplot(gs[0, 2])
    ax_overlay.imshow(overlay_disp)
    ax_overlay.axis("off")
    ax_overlay.set_title("Overlay")

    # Bar chart of probabilities
    ax_bar = fig.add_subplot(gs[1, :])
    labels = (
        class_names if class_names is not None else [str(i) for i in range(len(class_probs))]
    )
    ax_bar.bar(labels, class_probs, color="steelblue")
    ax_bar.set_ylabel("Probability")
    pred_label = labels[predicted_class]
    ax_bar.set_title(f"Predicted: {pred_label} ({class_probs[predicted_class]:.2%})")
    ax_bar.set_ylim(0, 1)
    ax_bar.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    if save_path:
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig

# End of file
