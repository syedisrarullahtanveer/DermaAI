"""
Immersion gel bubble detector and inpainting filter for dermoscopy images.
"""
import cv2
import numpy as np

class GelBubbleFilter:
    def filter_bubbles(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        # Gel bubbles present high circular specular reflection with dark rim
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=50, param2=30, minRadius=4, maxRadius=25
        )
        if circles is None:
            return image
        
        mask = np.zeros(gray.shape, dtype=np.uint8)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
            
        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted
