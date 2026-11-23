"""
Evidential deep learning for subjective Dirichlet uncertainty estimation.
Reference: Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty", NeurIPS 2018.
"""
import numpy as np

class EvidentialUncertaintyQuantifier:
    def compute_evidential_metrics(self, evidence: np.ndarray) -> dict:
        # evidence >= 0
        evidence = np.maximum(evidence, 0.0)
        alpha = evidence + 1.0
        s = np.sum(alpha, axis=-1, keepdims=True)
        # Expected probability
        belief = evidence / s
        uncertainty = evidence.shape[-1] / s
        return {
            "expected_prob": belief,
            "vacuity_uncertainty": uncertainty.squeeze(-1)
        }
