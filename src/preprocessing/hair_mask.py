"""Hair and Artifact Mask Generator for Dermoscopic Image Cleaning.

Uses morphological black top-hat filtering to isolate dark hair fibers
and generate binary masks suitable for OpenCV inpainting.
"""

from typing import Tuple
import cv2
import numpy as np


class HairMaskGenerator:
    """Generate binary masks segmenting hair artifacts from skin lesions."""

    def __init__(self, kernel_size: int = 9, threshold: int = 10) -> None:
        """Initialize hair mask generator.

        Args:
            kernel_size: Size of square structuring element for morphological top-hat (odd int).
            threshold: Intensity threshold for binarizing top-hat filtered hair fibers.
        """
        if kernel_size % 2 == 0 or kernel_size < 3:
            raise ValueError(f"kernel_size must be an odd integer >= 3, got {kernel_size}")
        self.kernel_size = kernel_size
        self.threshold = threshold
        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    def generate_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate binary hair mask from RGB or grayscale image.

        Args:
            image: numpy array (H, W, 3) or (H, W) in uint8 [0, 255].

        Returns:
            np.ndarray: Binary uint8 mask of shape (H, W) where 255 represents hair pixels.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("Image must be a numpy ndarray")
        if image.ndim not in (2, 3):
            raise ValueError(f"Expected 2D or 3D image, got {image.ndim}D")

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image.copy()
        gray = np.clip(gray, 0, 255).astype(np.uint8)

        # Black top-hat filtering isolates dark thin structures on brighter backgrounds
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self._kernel)

        # Threshold to create binary mask
        _, mask = cv2.threshold(top_hat, self.threshold, 255, cv2.THRESH_BINARY)
        return mask
