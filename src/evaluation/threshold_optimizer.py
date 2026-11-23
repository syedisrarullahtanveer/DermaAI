"""
Clinical Sensitivity-Specificity threshold optimizer for binary melanoma risk decisions.
"""
import numpy as np

class ClinicalThresholdOptimizer:
    """Find classification threshold targeting a minimum clinical sensitivity (e.g., 95% recall for melanoma)."""
    def __init__(self, target_sensitivity: float = 0.95):
        self.target_sensitivity = target_sensitivity

    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        thresholds = np.linspace(0.01, 0.99, 100)
        best_threshold = 0.5
        best_spec = 0.0
        actual_sens = 0.0

        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            tp = np.sum((y_true == 1) & (preds == 1))
            fn = np.sum((y_true == 1) & (preds == 0))
            tn = np.sum((y_true == 0) & (preds == 0))
            fp = np.sum((y_true == 0) & (preds == 1))
            
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            if sens >= self.target_sensitivity and spec > best_spec:
                best_spec = spec
                best_threshold = t
                actual_sens = sens

        return {"threshold": float(best_threshold), "sensitivity": float(actual_sens), "specificity": float(best_spec)}
