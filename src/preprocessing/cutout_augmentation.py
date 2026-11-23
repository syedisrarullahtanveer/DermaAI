"""
Random Cutout augmentation for DermaAI.

Randomly masks square patches in input images to improve robustness
to partial occlusions found in dermoscopy images (hair, artifacts).
Reference: DeVries & Taylor, "Improved Regularization of Convolutional
Neural Networks with Cutout", arXiv:1708.04552.
"""

import numpy as np


def random_cutout(
    image: np.ndarray,
    num_patches: int = 1,
    patch_size: int = 32,
    fill_value: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply random cutout augmentation to a single image.

    Args:
        image: Input image array of shape (H, W, C) with float values.
        num_patches: Number of cutout patches to apply.
        patch_size: Side length of each square patch (pixels).
        fill_value: Value used to fill the masked region.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        Augmented image with masked patches.
    """
    if rng is None:
        rng = np.random.default_rng()

    img = image.copy()
    h, w = img.shape[:2]
    half = patch_size // 2

    for _ in range(num_patches):
        cy = rng.integers(0, h)
        cx = rng.integers(0, w)
        y1 = max(0, cy - half)
        y2 = min(h, cy + half)
        x1 = max(0, cx - half)
        x2 = min(w, cx + half)
        img[y1:y2, x1:x2] = fill_value

    return img


class CutoutAugmentation:
    """Stateful Cutout augmentation that can be applied to batches.

    Args:
        num_patches: Number of patches per image.
        patch_size: Patch side length in pixels.
        fill_value: Fill value for masked regions (default 0 = black).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        num_patches: int = 1,
        patch_size: int = 32,
        fill_value: float = 0.0,
        seed: int | None = None,
    ):
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.fill_value = fill_value
        self._rng = np.random.default_rng(seed)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Apply cutout to a batch of images.

        Args:
            images: Array of shape (N, H, W, C) or (H, W, C).

        Returns:
            Augmented image(s) with same shape.
        """
        if images.ndim == 3:
            return random_cutout(images, self.num_patches, self.patch_size, self.fill_value, self._rng)
        return np.stack(
            [random_cutout(img, self.num_patches, self.patch_size, self.fill_value, self._rng)
             for img in images]
        )
