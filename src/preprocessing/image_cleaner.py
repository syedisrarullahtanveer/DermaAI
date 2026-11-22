"""DullRazor Hair Removal and Illumination Color Constancy Algorithms.

Removes occluding hairs, ruler marks, and lighting artifacts from dermoscopy images.
"""

from typing import Tuple, Union
import cv2
import numpy as np


class DullRazorCleaner:
    """Implementation of the DullRazor morphological filter and inpainting algorithm for hair removal."""

    def __init__(
        self,
        filter_kernel_size: int = 9,
        threshold: int = 10,
        inpaint_radius: int = 5,
    ):
        self.kernel_size = filter_kernel_size
        self.threshold = threshold
        self.inpaint_radius = inpaint_radius
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.kernel_size, self.kernel_size)
        )

    def remove_hair(self, image: np.ndarray) -> np.ndarray:
        """Apply morphological black-hat transform and Telea inpainting to eliminate occluding hair.

        Args:
            image: Input RGB image (numpy array, shape [H, W, 3], uint8 or float32).

        Returns:
            np.ndarray: Cleaned RGB image of same dimensions.
        """
        is_float = False
        if image.dtype in [np.float32, np.float64]:
            is_float = True
            img_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img_uint8 = image.copy()

        # Step 1: Convert to grayscale
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)

        # Step 2: Morphological Black-Hat transform to isolate dark linear structures (hairs)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, self.kernel)

        # Step 3: Adaptive thresholding to generate binary hair mask
        _, hair_mask = cv2.threshold(blackhat, self.threshold, 255, cv2.THRESH_BINARY)

        # Step 4: Telea Fast Marching inpainting over detected hair pixels
        cleaned_bgr = cv2.inpaint(
            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR),
            hair_mask,
            self.inpaint_radius,
            cv2.INPAINT_TELEA,
        )
        cleaned_rgb = cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB)

        if is_float:
            return cleaned_rgb.astype(np.float32) / 255.0
        return cleaned_rgb


class GrayWorldColorConstancy:
    """Gray World illumination normalization algorithm.

    Normalizes the average color of a dermoscopic scene to neutral gray,
    removing color cast caused by differing camera sensors or light sources.
    """

    @staticmethod
    def apply(image: np.ndarray) -> np.ndarray:
        """Apply Gray World color constancy adjustment.

        Args:
            image: Input RGB image.

        Returns:
            np.ndarray: Color-balanced RGB image.
        """
        is_float = False
        if image.dtype in [np.float32, np.float64]:
            is_float = True
            img = np.clip(image * 255.0, 0, 255).astype(np.float32)
        else:
            img = image.astype(np.float32)

        # Compute per-channel mean
        r_mean = np.mean(img[:, :, 0])
        g_mean = np.mean(img[:, :, 1])
        b_mean = np.mean(img[:, :, 2])

        gray_mean = (r_mean + g_mean + b_mean) / 3.0

        # Avoid division by zero
        r_scale = gray_mean / (r_mean + 1e-6)
        g_scale = gray_mean / (g_mean + 1e-6)
        b_scale = gray_mean / (b_mean + 1e-6)

        img[:, :, 0] = np.clip(img[:, :, 0] * r_scale, 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * g_scale, 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * b_scale, 0, 255)

        if is_float:
            return (img / 255.0).astype(np.float32)
        return img.astype(np.uint8)
