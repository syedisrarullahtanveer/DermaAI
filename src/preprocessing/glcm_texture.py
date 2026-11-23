"""
Gray-Level Co-occurrence Matrix (GLCM) texture feature extractor for lesion surface granularity.
Reference: Haralick et al., "Textural Features for Image Classification", IEEE SMC 1973.
"""
import numpy as np

class GLCMTextureExtractor:
    def __init__(self, distances=(1,), angles=(0,)):
        self.distances = distances
        self.angles = angles

    def compute_contrast(self, gray_img: np.ndarray) -> float:
        # Simplified horizontal adjacent co-occurrence contrast
        diff = np.diff(gray_img.astype(float), axis=1)
        return float(np.mean(diff ** 2))

    def compute_homogeneity(self, gray_img: np.ndarray) -> float:
        diff = np.abs(np.diff(gray_img.astype(float), axis=1))
        return float(np.mean(1.0 / (1.0 + diff)))
