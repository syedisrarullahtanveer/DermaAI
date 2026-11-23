"""Dermoscopic Lens Vignette and Dark Circular Border Removal.

Removes dark corner artifacts caused by dermatoscope optics and cylindrical contact plates.
"""

from typing import Tuple, Union
import cv2
import numpy as np


class VignetteCornerRemover:
    """Detect and remove dark circular vignettes/black borders in dermoscopy images."""

    def __init__(self, intensity_threshold: int = 25, min_radius_ratio: float = 0.35) -> None:
        """Initialize vignette remover.

        Args:
            intensity_threshold: Pixel brightness threshold below which corners are considered border.
            min_radius_ratio: Minimum fraction of half-diagonal for bounding circle.
        """
        self.intensity_threshold = intensity_threshold
        self.min_radius_ratio = min_radius_ratio

    def crop_vignette(self, image: np.ndarray) -> np.ndarray:
        """Crop or mask the circular aperture to eliminate dark peripheral artifacts.

        Args:
            image: numpy array (H, W, 3) or (H, W) in uint8 [0, 255].

        Returns:
            np.ndarray: Cropped or border-cleaned image.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy ndarray")
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image, got shape: {image.shape}")

        gray = image.mean(axis=2).astype(np.uint8) if image.ndim == 3 else image.copy()
        h, w = gray.shape

        # Mask pixels above dark border threshold
        mask = (gray > self.intensity_threshold).astype(np.uint8)

        # Find connected components or contours to identify central illuminated circle
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image.copy()

        # Largest contour corresponds to illuminated field of view
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, cw, ch = cv2.boundingRect(largest_contour)

        # Ensure bounding box is valid and significant
        if cw < w * self.min_radius_ratio or ch < h * self.min_radius_ratio:
            return image.copy()

        cropped = image[y : y + ch, x : x + cw]
        return cropped
