"""
CutBlur data augmentation: pastes a downscaled and blurred sub-patch onto a high-resolution dermoscopic image.
Reference: Yoo et al., "Rethinking Data Augmentation for Image Super-resolution: A Comprehensive Analysis and a New Strategy", CVPR 2020.
"""
import cv2
import numpy as np

class CutBlurAugmentation:
    def __init__(self, blur_kernel: int = 5, downscale_factor: int = 2):
        self.blur_kernel = blur_kernel
        self.downscale_factor = downscale_factor

    def apply(self, image: np.ndarray, bbox: tuple) -> np.ndarray:
        x, y, w, h = bbox
        out = image.copy()
        patch = out[y:y+h, x:x+w]
        if patch.size == 0:
            return out
        # Downscale and blur
        small = cv2.resize(patch, (max(1, w // self.downscale_factor), max(1, h // self.downscale_factor)))
        restored = cv2.resize(small, (w, h))
        blurred = cv2.GaussianBlur(restored, (self.blur_kernel, self.blur_kernel), 0)
        out[y:y+h, x:x+w] = blurred
        return out
