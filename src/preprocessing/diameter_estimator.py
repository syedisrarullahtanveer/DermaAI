"""
Lesion diameter estimator (D in ABCD criteria) with physical millimeter scaling.
"""
import cv2
import numpy as np

class LesionDiameterEstimator:
    def __init__(self, mm_per_pixel: float = 0.05):
        self.mm_per_pixel = mm_per_pixel

    def estimate_diameter(self, binary_mask: np.ndarray) -> dict:
        contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {"diameter_pixels": 0.0, "diameter_mm": 0.0, "exceeds_6mm": False}
        c = max(contours, key=cv2.contourArea)
        # Minimum enclosing circle
        (_, _), radius = cv2.minEnclosingCircle(c)
        diameter_px = radius * 2.0
        diameter_mm = diameter_px * self.mm_per_pixel
        return {
            "diameter_pixels": float(diameter_px),
            "diameter_mm": float(diameter_mm),
            "exceeds_6mm": bool(diameter_mm >= 6.0)
        }
