"""
Local Contrast Normalization (LCN) across local Gaussian sliding windows.
"""
import cv2
import numpy as np

class LocalContrastNormalizer:
    def normalize(self, image: np.ndarray, ksize: int = 9) -> np.ndarray:
        img_f = image.astype(np.float32)
        local_mean = cv2.GaussianBlur(img_f, (ksize, ksize), 0)
        diff = img_f - local_mean
        local_var = cv2.GaussianBlur(diff ** 2, (ksize, ksize), 0)
        local_std = np.sqrt(np.maximum(local_var, 1e-4))
        normed = diff / local_std
        return normed
