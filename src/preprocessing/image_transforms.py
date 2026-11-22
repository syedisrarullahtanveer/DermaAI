"""Advanced Image Transformations: Resizing, Letterbox Padding, and Normalization."""

from typing import List, Optional, Tuple, Union
import cv2
import numpy as np


class ImageResizer:
    """Handles spatial resizing with options for direct scaling, center-cropping, and aspect-ratio letterboxing."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        mode: str = "letterbox",
        interpolation: int = cv2.INTER_AREA,
        pad_value: int = 0,
    ):
        self.target_size = target_size
        self.mode = mode.lower()
        self.interpolation = interpolation
        self.pad_value = pad_value

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image according to selected mode ('direct', 'letterbox', 'center_crop').

        Args:
            image: Input RGB image of shape [H, W, C].

        Returns:
            np.ndarray: Resized RGB image of shape [target_H, target_W, C].
        """
        target_h, target_w = self.target_size

        if self.mode == "direct":
            return cv2.resize(image, (target_w, target_h), interpolation=self.interpolation)

        elif self.mode == "center_crop":
            h, w = image.shape[:2]
            # First scale so smallest dimension matches target
            scale = max(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            scaled = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

            # Center crop
            start_x = (new_w - target_w) // 2
            start_y = (new_h - target_h) // 2
            return scaled[start_y : start_y + target_h, start_x : start_x + target_w]

        elif self.mode == "letterbox":
            h, w = image.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)

            scaled = cv2.resize(image, (new_w, new_h), interpolation=self.interpolation)

            # Create padded canvas
            channels = image.shape[2] if image.ndim == 3 else 1
            if channels == 1:
                canvas = np.full((target_h, target_w), self.pad_value, dtype=image.dtype)
            else:
                canvas = np.full((target_h, target_w, channels), self.pad_value, dtype=image.dtype)

            # Compute offsets
            offset_x = (target_w - new_w) // 2
            offset_y = (target_h - new_h) // 2

            canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = scaled
            return canvas

        else:
            raise ValueError(f"Unknown resize mode: '{self.mode}'. Supported: 'direct', 'letterbox', 'center_crop'.")


class ImageNormalizer:
    """Applies statistical normalization pipelines across dermoscopic images."""

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, method: str = "min_max"):
        self.method = method.lower()

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image according to designated method ('min_max', 'zero_center', 'imagenet', 'standardize')."""
        img_float = image.astype(np.float32)

        # Scale to [0, 1] if uint8
        if image.dtype == np.uint8:
            img_float = img_float / 255.0

        if self.method == "min_max":
            # Values in [0, 1]
            return np.clip(img_float, 0.0, 1.0)

        elif self.method == "zero_center":
            # Values in [-1, 1]
            return (img_float * 2.0) - 1.0

        elif self.method == "imagenet":
            # Per-channel z-score standardization against ImageNet statistics
            return (img_float - self.IMAGENET_MEAN) / self.IMAGENET_STD

        elif self.method == "standardize":
            # Per-image mean-zero unit-variance
            mean = np.mean(img_float, axis=(0, 1), keepdims=True)
            std = np.std(img_float, axis=(0, 1), keepdims=True) + 1e-7
            return (img_float - mean) / std

        else:
            raise ValueError(f"Unknown normalization method: '{self.method}'")

    def denormalize(self, normalized_image: np.ndarray) -> np.ndarray:
        """Invert normalization back to standard uint8 RGB image [0, 255]."""
        if self.method == "min_max":
            denorm = normalized_image * 255.0

        elif self.method == "zero_center":
            denorm = ((normalized_image + 1.0) / 2.0) * 255.0

        elif self.method == "imagenet":
            denorm = (normalized_image * self.IMAGENET_STD + self.IMAGENET_MEAN) * 255.0

        elif self.method == "standardize":
            # Approximate inverse
            min_val = np.min(normalized_image)
            max_val = np.max(normalized_image)
            denorm = ((normalized_image - min_val) / (max_val - min_val + 1e-7)) * 255.0

        else:
            denorm = normalized_image * 255.0

        return np.clip(denorm, 0, 255).astype(np.uint8)
