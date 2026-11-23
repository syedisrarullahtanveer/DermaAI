"""
LayerCAM: generates fine-grained class activation maps from intermediate convolutional layers.
Reference: Jiang et al., "LayerCAM: Exploring Hierarchical Class Activation Maps for Localization", TIP 2021.
"""
import numpy as np

class LayerCAM:
    def compute_cam(self, activations: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        # activations: (H, W, C), gradients: (H, W, C)
        # Element-wise positive gradient weighting
        weights = np.maximum(gradients, 0)
        weighted_act = weights * activations
        cam = np.sum(weighted_act, axis=-1)
        cam = np.maximum(cam, 0)
        max_val = np.max(cam)
        if max_val > 0:
            cam /= max_val
        return cam
