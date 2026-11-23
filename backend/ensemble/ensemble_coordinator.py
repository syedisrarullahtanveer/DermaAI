"""
Multi-model ensemble inference coordinator (EdgeCNN + ResNet50 + EfficientNet) with weighted Bayesian voting.
"""
import numpy as np

class EnsembleCoordinator:
    def __init__(self, weights: dict | None = None):
        # Default equal weighting if none specified
        self.weights = weights or {"edge_cnn": 0.3, "efficientnet": 0.4, "resnet50": 0.3}

    def combine_predictions(self, model_outputs: dict) -> np.ndarray:
        final_probs = None
        total_weight = 0.0
        
        for name, probs in model_outputs.items():
            w = self.weights.get(name, 0.0)
            p = np.asarray(probs)
            if final_probs is None:
                final_probs = w * p
            else:
                final_probs += w * p
            total_weight += w
            
        if total_weight > 0:
            final_probs /= total_weight
        return final_probs
