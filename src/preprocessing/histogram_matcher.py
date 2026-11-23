"""
Color histogram matching for cross-center dermoscopic color standardization.
"""
import numpy as np

class HistogramMatcher:
    def match_channel(self, source: np.ndarray, template: np.ndarray) -> np.ndarray:
        s_values, bin_idx, s_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(template.ravel(), return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64) / source.size
        t_quantiles = np.cumsum(t_counts).astype(np.float64) / template.size
        interp_values = np.interp(s_quantiles, t_quantiles, t_values)
        return interp_values[bin_idx].reshape(source.shape).astype(source.dtype)

    def match(self, source_img: np.ndarray, template_img: np.ndarray) -> np.ndarray:
        matched = np.empty_like(source_img)
        for c in range(source_img.shape[-1]):
            matched[..., c] = self.match_channel(source_img[..., c], template_img[..., c])
        return matched
