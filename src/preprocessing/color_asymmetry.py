"""
Color asymmetry computation across major and minor geometric axes (A in ABCD criteria).
"""
import numpy as np

class ColorAsymmetryComputer:
    def compute_asymmetry(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        mid_y, mid_x = h // 2, w // 2
        # Vertical asymmetry
        top = image[:mid_y, :]
        bottom = np.flipud(image[h - mid_y:, :])
        vert_diff = float(np.mean(np.abs(top.astype(float) - bottom.astype(float))))

        # Horizontal asymmetry
        left = image[:, :mid_x]
        right = np.fliplr(image[:, w - mid_x:])
        horiz_diff = float(np.mean(np.abs(left.astype(float) - right.astype(float))))

        return {"horizontal_asymmetry": horiz_diff, "vertical_asymmetry": vert_diff}
