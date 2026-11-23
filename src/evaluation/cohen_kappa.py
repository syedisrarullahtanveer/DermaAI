"""
Cohen's Kappa evaluator for clinical inter-rater agreement and multiclass evaluation.
"""
import numpy as np
from sklearn.metrics import cohen_kappa_score

class CohensKappaEvaluator:
    """Compute quadratic or linear weighted Cohen's Kappa score for diagnostic consensus."""
    def __init__(self, weights: str | None = "quadratic"):
        self.weights = weights

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim > 1:
            y_true = np.argmax(y_true, axis=-1)
        if y_pred.ndim > 1:
            y_pred = np.argmax(y_pred, axis=-1)
        return float(cohen_kappa_score(y_true, y_pred, weights=self.weights))
