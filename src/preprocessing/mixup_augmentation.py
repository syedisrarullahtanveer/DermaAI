"""
MixUp augmentation for dermoscopy images in DermaAI.

Blends pairs of training images and their labels by convex interpolation,
improving calibration and reducing memorization.
Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018.
"""

import numpy as np


def mixup_pair(
    image_a: np.ndarray,
    label_a: np.ndarray,
    image_b: np.ndarray,
    label_b: np.ndarray,
    alpha: float = 0.4,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix two image-label pairs using a Beta-distributed interpolation factor.

    Args:
        image_a: First image (H, W, C) float array.
        label_a: One-hot label for image_a (num_classes,).
        image_b: Second image (H, W, C) float array.
        label_b: One-hot label for image_b (num_classes,).
        alpha: Beta distribution concentration parameter. Higher values
               produce more mixing; alpha=0 disables mixing.
        rng: Optional NumPy random generator.

    Returns:
        (mixed_image, mixed_label) tuple.
    """
    if rng is None:
        rng = np.random.default_rng()

    if alpha > 0:
        lam = rng.beta(alpha, alpha)
    else:
        lam = 1.0

    mixed_image = lam * image_a + (1.0 - lam) * image_b
    mixed_label = lam * label_a + (1.0 - lam) * label_b
    return mixed_image.astype(np.float32), mixed_label.astype(np.float32)


def mixup_batch(
    images: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.4,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MixUp to an entire batch by shuffling and mixing with itself.

    Args:
        images: Batch of images (N, H, W, C).
        labels: Batch of one-hot labels (N, num_classes).
        alpha: Beta distribution concentration parameter.
        seed: Random seed.

    Returns:
        (mixed_images, mixed_labels) with same shape as inputs.
    """
    rng = np.random.default_rng(seed)
    n = images.shape[0]
    indices = rng.permutation(n)
    mixed_images = np.empty_like(images)
    mixed_labels = np.empty_like(labels)
    for i in range(n):
        mi, ml = mixup_pair(images[i], labels[i], images[indices[i]], labels[indices[i]], alpha, rng)
        mixed_images[i] = mi
        mixed_labels[i] = ml
    return mixed_images, mixed_labels
