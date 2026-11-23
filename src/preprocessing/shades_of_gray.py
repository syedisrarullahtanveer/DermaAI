"""Shades of Gray Color Constancy Preprocessing for Dermoscopic Images.

Reference:
    Finlayson, G. D., & Trezzi, E. (2004). Shades of Gray: Colour Constancy
    Using a General Minkowski Norm. In IS&T/SID Color and Imaging Conference.
"""

from typing import Union
import numpy as np


class ShadesOfGrayConstancy:
    """Normalize illumination variations across dermoscopy images using Minkowski p-norm."""

    def __init__(self, power: int = 6) -> None:
        """Initialize Shades of Gray transformer.

        Args:
            power: Minkowski norm power parameter (p=1: Gray-World, p=inf: Max-RGB, p=6: Shades of Gray).
        """
        if power < 1:
            raise ValueError(f"Power parameter must be >= 1, received: {power}")
        self.power = power

    def transform(self, image: np.ndarray) -> np.ndarray:
        """Apply Shades of Gray color constancy to an RGB image.

        Args:
            image: Input image array of shape (H, W, 3) with values in [0, 255] or [0.0, 1.0].

        Returns:
            np.ndarray: Color-normalized image with preserved dtype and value range.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected 3-channel RGB image (H, W, 3), got shape: {image.shape}")

        is_integer = np.issubdtype(image.dtype, np.integer)
        float_img = image.astype(np.float64)

        # Compute Minkowski p-norm estimate of the illuminant for each channel
        p = self.power
        illuminant = np.power(np.mean(np.power(float_img, p), axis=(0, 1)), 1.0 / p)

        # Avoid division by zero
        illuminant_norm = np.linalg.norm(illuminant)
        if illuminant_norm < 1e-8:
            return image.copy()

        # Compute target neutral gray vector
        target_scale = illuminant_norm / np.sqrt(3.0)
        scale_factors = target_scale / (illuminant + 1e-8)

        normalized = float_img * scale_factors
        max_bound = 255.0 if is_integer else 1.0
        normalized = np.clip(normalized, 0.0, max_bound)

        if is_integer:
            return np.round(normalized).astype(image.dtype)
        return normalized.astype(image.dtype)
