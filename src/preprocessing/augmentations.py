"""Dermoscopy-Specific Data Augmentation Pipeline."""

from typing import Tuple
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


class DermaAugmentationPipeline:
    """Applies clinical dermoscopic augmentations: affine rotations, flips, zoom, and contrast shifts."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def augment_tf_tensor(self, image, label):
        """TensorFlow graph augmentation function for tf.data pipeline mapping."""
        if tf is None:
            return image, label

        # Random horizontal and vertical flips (dermoscopy lesions are rotational invariant)
        image = tf.image.random_flip_left_right(image, seed=self.seed)
        image = tf.image.random_flip_up_down(image, seed=self.seed)

        # Random brightness and contrast adjustments
        image = tf.image.random_brightness(image, max_delta=0.15, seed=self.seed)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15, seed=self.seed)

        # Ensure values stay clamped in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    @staticmethod
    def mixup(
        image1: np.ndarray,
        image2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray,
        alpha: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply MixUp regularization across two lesion samples and their one-hot labels."""
        lam = np.random.beta(alpha, alpha)
        mixed_image = lam * image1 + (1.0 - lam) * image2
        mixed_label = lam * label1 + (1.0 - lam) * label2
        return mixed_image, mixed_label
