"""
Border irregularity (B in ABCD criteria) metric for dermoscopic lesion segmentation masks.
"""
import cv2
import numpy as np

class BorderIrregularityExtractor:
    def compute_compactness(self, binary_mask: np.ndarray) -> float:
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        if area == 0 or perimeter == 0:
            return 0.0
        # Isoperimetric quotient: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
        score = (4 * np.pi * area) / (perimeter ** 2)
        return float(score)
