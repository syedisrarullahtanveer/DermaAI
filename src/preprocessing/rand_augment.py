"""
RandAugment policy tailored for dermoscopy images.
Reference: Cubuk et al., "RandAugment: Practical automated data augmentation", CVPRW 2020.
"""
import cv2
import numpy as np

class RandAugmentDermoscopy:
    def __init__(self, n: int = 2, m: int = 5, seed: int = 42):
        self.n = n
        self.m = m
        self.rng = np.random.default_rng(seed)

    def _rotate(self, img, m):
        angle = (m / 10.0) * 45.0 * (1 if self.rng.random() > 0.5 else -1)
        h, w = img.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _brightness(self, img, m):
        factor = 1.0 + (m / 10.0) * 0.4 * (1 if self.rng.random() > 0.5 else -1)
        return np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)

    def augment(self, image: np.ndarray) -> np.ndarray:
        ops = [self._rotate, self._brightness]
        out = image.copy()
        chosen = self.rng.choice(ops, size=self.n, replace=True)
        for op in chosen:
            out = op(out, self.m)
        return out
