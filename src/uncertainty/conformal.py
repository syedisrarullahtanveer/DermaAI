"""
Split-conformal prediction sets with finite-sample coverage guarantee.
Reference: Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", 2021.
"""
import numpy as np

class ConformalPredictionEvaluator:
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.q_hat = 1.0

    def calibrate(self, cal_probs: np.ndarray, cal_labels: np.ndarray):
        n = len(cal_labels)
        # Non-conformity scores: 1 - prob[true_class]
        true_class_probs = cal_probs[np.arange(n), cal_labels]
        scores = 1.0 - true_class_probs
        # Quantile
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        self.q_hat = float(np.quantile(scores, min(1.0, level)))
        return self.q_hat

    def predict_sets(self, test_probs: np.ndarray) -> list:
        # Include classes whose probability >= 1 - q_hat
        threshold = 1.0 - self.q_hat
        return [list(np.where(p >= threshold)[0]) for p in test_probs]
