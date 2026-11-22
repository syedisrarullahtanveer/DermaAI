"""Heatmap Overlay and Visual Inspection Module for Explainable AI."""

from typing import Tuple
import cv2
import numpy as np


class HeatmapVisualizer:
    """Superimposes Grad-CAM heatmaps onto dermoscopy images with configurable opacity and colormaps."""

    @staticmethod
    def overlay_heatmap(
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """Blend a 2D activation heatmap over an RGB dermoscopic image.

        Args:
            image: Original RGB image array [H, W, 3] with values in [0, 255] or [0, 1].
            heatmap: 2D normalized activation map [h, w] with values in [0, 1].
            alpha: Transparency factor of the colored heatmap overlay (0.0 to 1.0).
            colormap: OpenCV colormap constant (default: COLORMAP_JET).

        Returns:
            np.ndarray: Blended RGB image of uint8 format [H, W, 3].
        """
        # Ensure image is uint8 in range [0, 255]
        if image.dtype in [np.float32, np.float64]:
            orig_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            orig_uint8 = image.copy()

        h, w = orig_uint8.shape[:2]

        # Resize heatmap to match image spatial resolution
        resized_cam = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        cam_uint8 = np.uint8(255 * np.clip(resized_cam, 0, 1))

        # Apply colormap to generate colored thermal overlay (BGR)
        colored_bgr = cv2.applyColorMap(cam_uint8, colormap)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)

        # Alpha blend original image with thermal map
        blended = cv2.addWeighted(orig_uint8, 1.0 - alpha, colored_rgb, alpha, 0)
        return blended

    @staticmethod
    def extract_salient_contour(
        heatmap: np.ndarray,
        target_size: Tuple[int, int],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Extract boundary contours of the most salient lesion regions indicated by the heatmap."""
        resized_cam = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_LINEAR)
        binary_mask = (resized_cam >= threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
