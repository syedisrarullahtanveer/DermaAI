"""
JPEG compression artifact detector and blockiness scoring for dermoscopic images.
"""
import numpy as np

class JPEGArtifactDetector:
    def score_blockiness(self, gray_image: np.ndarray) -> float:
        img = np.asarray(gray_image, dtype=np.float32)
        # Difference across 8x8 block boundaries vs within-block differences
        diff_h = np.abs(img[7::8, :] - img[8::8, :])
        diff_v = np.abs(img[:, 7::8] - img[:, 8::8])
        blockiness = (np.mean(diff_h) + np.mean(diff_v)) / 2.0
        return float(blockiness)
