"""
Effective number of samples reweighting calculator for severe class imbalance.
Reference: Cui et al., "Class-Balanced Loss Based on Effective Number of Samples", CVPR 2019.
"""
import numpy as np

class ClassWeightCalculator:
    def __init__(self, beta: float = 0.999):
        self.beta = beta

    def compute_weights(self, class_counts: list | np.ndarray) -> dict:
        counts = np.array(class_counts, dtype=np.float64)
        effective_num = 1.0 - np.power(self.beta, counts)
        weights = (1.0 - self.beta) / (effective_num + 1e-8)
        # Normalize so weights sum to num_classes
        weights = weights / np.sum(weights) * len(counts)
        return {i: float(w) for i, w in enumerate(weights)}
