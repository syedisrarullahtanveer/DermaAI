"""Macenko Optical Density Stain and Pigment Normalization for Skin Lesions.

Reference:
    Macenko, M., et al. (2009). A method for normalizing histology slides
    for quantitative analysis. IEEE ISBI 2009.
"""

from typing import Tuple
import numpy as np


class MacenkoNormalizer:
    """Normalize pigment optical density variations using SVD decomposition."""

    def __init__(self, beta: float = 0.15, alpha: float = 1.0) -> None:
        """Initialize Macenko normalizer.

        Args:
            beta: OD threshold for transparent/background pixel exclusion.
            alpha: Percentile tolerance for robust angle detection.
        """
        self.beta = beta
        self.alpha = alpha

    def rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB image in [0, 255] to Optical Density (OD) space."""
        image = np.maximum(image.astype(np.float64), 1.0)
        return -np.log10(image / 255.0)

    def od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert Optical Density (OD) back to RGB in [0, 255]."""
        rgb = 255.0 * np.power(10.0, -od)
        return np.clip(np.round(rgb), 0, 255).astype(np.uint8)

    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize optical density across color absorption channels.

        Args:
            image: uint8 RGB numpy array of shape (H, W, 3).

        Returns:
            np.ndarray: Normalized RGB uint8 image.
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected uint8 RGB array of shape (H, W, 3)")

        od = self.rgb_to_od(image)
        od_flat = od.reshape(-1, 3)

        # Exclude pixels with optical density below beta
        mask = np.any(od_flat > self.beta, axis=1)
        od_valid = od_flat[mask]

        if len(od_valid) < 10:
            return image.copy()

        # Compute eigenvectors using SVD of covariance matrix
        _, _, vh = np.linalg.svd(od_valid, full_matrices=False)
        # Primary plane spanned by first two eigenvectors
        plane = vh[:2]

        # Project valid OD pixels onto plane
        projections = np.dot(od_valid, plane.T)
        angles = np.arctan2(projections[:, 1], projections[:, 0])

        min_angle = np.percentile(angles, self.alpha)
        max_angle = np.percentile(angles, 100.0 - self.alpha)

        v_min = np.dot(plane.T, np.array([np.cos(min_angle), np.sin(min_angle)]))
        v_max = np.dot(plane.T, np.array([np.cos(max_angle), np.sin(max_angle)]))

        stain_matrix = np.stack([v_min, v_max], axis=1)
        stain_matrix = stain_matrix / (np.linalg.norm(stain_matrix, axis=0, keepdims=True) + 1e-8)

        # Reconstruct normalized optical density
        concentrations, _, _, _ = np.linalg.lstsq(stain_matrix, od_flat.T, rcond=None)
        reconstructed_od = np.dot(stain_matrix, concentrations).T
        reconstructed_od = reconstructed_od.reshape(image.shape)

        return self.od_to_rgb(reconstructed_od)
