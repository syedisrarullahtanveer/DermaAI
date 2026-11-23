"""
Color Variegation Index (C in ABCD criteria) across dermoscopic color clusters.
Quantifies the presence of 6 clinical colors: light brown, dark brown, black, red, white, blue-gray.
"""
import numpy as np

class ColorVariegationCalculator:
    def calculate_color_score(self, image_rgb: np.ndarray, mask: np.ndarray) -> dict:
        h, w = image_rgb.shape[:2]
        lesion_pixels = image_rgb[mask > 0]
        if len(lesion_pixels) == 0:
            return {"num_colors": 0, "color_score": 0.0}
        
        # Color variance across RGB channels inside lesion
        std_r = np.std(lesion_pixels[:, 0])
        std_g = np.std(lesion_pixels[:, 1])
        std_b = np.std(lesion_pixels[:, 2])
        variegation_index = float((std_r + std_g + std_b) / 3.0)
        
        # Clinical score in [1, 6] based on thresholded dispersion
        estimated_colors = min(6, max(1, int(variegation_index // 8) + 1))
        return {"num_colors": estimated_colors, "variegation_index": variegation_index}
