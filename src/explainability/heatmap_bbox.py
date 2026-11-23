"""
Extract bounding boxes from Grad-CAM activation heatmaps for lesion localization.
"""
import cv2
import numpy as np

class HeatmapBoundingBoxExtractor:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def extract_bounding_box(self, heatmap: np.ndarray) -> tuple | None:
        binary = (heatmap >= self.threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return int(x), int(y), int(w), int(h)
