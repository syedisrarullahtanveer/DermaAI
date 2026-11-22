"""Comprehensive Data Augmentation Engine for Dermoscopy Lesion Classification."""

import random
from typing import Optional, Tuple
import cv2
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None


class DermaAugmentationPipeline:
    """Provides photometric, geometric, and batch-level regularizing augmentations for dermoscopy images."""

    def __init__(
        self,
        enable_flips: bool = True,
        enable_rotations: bool = True,
        enable_contrast_jitter: bool = True,
        enable_clahe: bool = True,
        enable_coarse_dropout: bool = True,
        seed: int = 42,
    ):
        self.enable_flips = enable_flips
        self.enable_rotations = enable_rotations
        self.enable_contrast_jitter = enable_contrast_jitter
        self.enable_clahe = enable_clahe
        self.enable_coarse_dropout = enable_coarse_dropout
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def augment_numpy(self, image: np.ndarray) -> np.ndarray:
        """Apply comprehensive augmentation sequence to a single RGB numpy image array.

        Args:
            image: Input RGB image [H, W, 3] in [0, 255] or [0, 1].

        Returns:
            np.ndarray: Augmented image with identical dtype and shape.
        """
        is_float = image.dtype in [np.float32, np.float64]
        if is_float:
            img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            img = image.copy()

        h, w = img.shape[:2]

        # 1. Random Flips
        if self.enable_flips:
            if random.random() > 0.5:
                img = cv2.flip(img, 1) # Horizontal flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0) # Vertical flip

        # 2. Random 90, 180, 270 degree Rotations (dermoscopy is strictly rotation invariant)
        if self.enable_rotations:
            rot_k = random.choice([0, 1, 2, 3])
            if rot_k > 0:
                img = np.rot90(img, k=rot_k)

        # 3. CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel in LAB space
        if self.enable_clahe and random.random() > 0.5:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        # 4. Brightness & Contrast Adjustment
        if self.enable_contrast_jitter and random.random() > 0.5:
            alpha = random.uniform(0.85, 1.15) # Contrast
            beta = random.uniform(-15, 15)      # Brightness
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # 5. Coarse Dropout / Random Erasing (simulates small pen marks or occlusion)
        if self.enable_coarse_dropout and random.random() > 0.6:
            drop_h = int(h * random.uniform(0.05, 0.15))
            drop_w = int(w * random.uniform(0.05, 0.15))
            y1 = random.randint(0, h - drop_h)
            x1 = random.randint(0, w - drop_w)
            # Fill with neutral gray or mean color
            img[y1 : y1 + drop_h, x1 : x1 + drop_w, :] = 128

        if is_float:
            return (img / 255.0).astype(image.dtype)
        return img

    def augment_tf_tensor(self, image, label):
        """Graph-compatible augmentation mapping function for tf.data pipelines."""
        if tf is None:
            return image, label

        # Rotational invariance
        image = tf.image.random_flip_left_right(image, seed=self.seed)
        image = tf.image.random_flip_up_down(image, seed=self.seed)

        # Lighting jitter
        image = tf.image.random_brightness(image, max_delta=0.15, seed=self.seed)
        image = tf.image.random_contrast(image, lower=0.85, upper=1.15, seed=self.seed)

        # Ensure range [0, 1]
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
        """MixUp data augmentation: linear interpolation between two lesion samples."""
        lam = np.random.beta(alpha, alpha)
        mixed_img = lam * image1 + (1.0 - lam) * image2
        mixed_lbl = lam * label1 + (1.0 - lam) * label2
        return mixed_img.astype(image1.dtype), mixed_lbl

    @staticmethod
    def cutmix(
        image1: np.ndarray,
        image2: np.ndarray,
        label1: np.ndarray,
        label2: np.ndarray,
        alpha: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """CutMix data augmentation: patch replacement between two lesion samples."""
        h, w = image1.shape[:2]
        lam = np.random.beta(alpha, alpha)

        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # Uniform random center of bounding box
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        mixed_img = image1.copy()
        mixed_img[bby1:bby2, bbx1:bbx2, :] = image2[bby1:bby2, bbx1:bbx2, :]

        # Adjust lambda to exact cut area proportion
        adjusted_lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        mixed_lbl = adjusted_lam * label1 + (1.0 - adjusted_lam) * label2

        return mixed_img, mixed_lbl
