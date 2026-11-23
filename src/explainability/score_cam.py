"""
Score-CAM gradient-free visual explainability.
Reference: Wang et al., "Score-CAM: Score-Weighted Visual Explanations for CNNs", CVPRW 2020.
"""
import cv2
import numpy as np

class ScoreCAM:
    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer_name = target_layer_name

    def generate_heatmap(self, activation_maps: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Linearly combine activation maps weighted by forward class score differences."""
        # activation_maps: (H, W, C), weights: (C,)
        cam = np.zeros(activation_maps.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activation_maps[..., i]
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam /= max_val
        return cam
