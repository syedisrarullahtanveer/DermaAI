"""Contrast Limited Adaptive Histogram Equalization (CLAHE) for Dermoscopy Images.

Operates in CIE LAB color space by enhancing the L (luminance) channel
while preserving authentic A and B color-opponent information.
"""

from typing import Tuple, Union
import cv2
import numpy as np


class CLAHEEnhancer:
    """Enhance skin lesion border and pigment contrast using adaptive histogram equalization."""

    def __init__(self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> None:
        """Initialize CLAHE enhancer.

        Args:
            clip_limit: Threshold for contrast limiting (higher values yield more contrast).
            tile_grid_size: Size of grid for local histogram calculation (width, height).
        """
        if clip_limit <= 0:
            raise ValueError(f"clip_limit must be > 0, got {clip_limit}")
        if len(tile_grid_size) != 2 or tile_grid_size[0] <= 0 or tile_grid_size[1] <= 0:
            raise ValueError(f"tile_grid_size must be a tuple of positive integers, got {tile_grid_size}")

        self.clip_limit = float(clip_limit)
        self.tile_grid_size = tile_grid_size
        self._clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)

    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE to an RGB or grayscale dermoscopy image.

        Args:
            image: numpy array of shape (H, W, 3) or (H, W) in uint8 [0, 255] or float [0.0, 1.0].

        Returns:
            np.ndarray: Contrast-enhanced image with original dtype and shape.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Input must be a numpy ndarray")
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image array, got {image.ndim}D")

        is_float = np.issubdtype(image.dtype, np.floating)
        if is_float:
            uint8_img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            uint8_img = np.clip(image, 0, 255).astype(np.uint8)

        if uint8_img.ndim == 2:
            enhanced_l = self._clahe.apply(uint8_img)
            result = enhanced_l
        else:
            # Convert RGB to CIE LAB color space
            lab = cv2.cvtColor(uint8_img, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            enhanced_l = self._clahe.apply(l_channel)
            merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            result = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)

        if is_float:
            return (result / 255.0).astype(image.dtype)
        return result.astype(image.dtype)
