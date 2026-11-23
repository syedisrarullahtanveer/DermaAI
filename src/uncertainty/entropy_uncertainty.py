"""
Shannon entropy and normalized uncertainty metrics for predicted probability vectors.
"""
import numpy as np

class PredictiveEntropyQuantifier:
    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def compute_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities), self.eps, 1.0)
        p = p / np.sum(p, axis=-1, keepdims=True)
        entropy = -np.sum(p * np.log(p), axis=-1)
        return entropy

    def compute_normalized_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        entropy = self.compute_entropy(probabilities)
        num_classes = probabilities.shape[-1]
        max_entropy = np.log(num_classes)
        return entropy / max_entropy
