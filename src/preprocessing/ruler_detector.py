"""
Detects and generates exclusion masks for ruler tick marks and calibration markers.
"""
import cv2
import numpy as np

class RulerArtifactDetector:
    def detect_ruler_marks(self, gray_image: np.ndarray) -> np.ndarray:
        # Rulers exhibit periodic high-contrast vertical or horizontal ticks
        edges = cv2.Canny(gray_image, 100, 200)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        return vertical_lines
