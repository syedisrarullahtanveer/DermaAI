"""
Cost-sensitive clinical risk matrix evaluator.
"""
import numpy as np

class CostSensitiveEvaluator:
    """Evaluates clinical misclassification penalty (e.g. missing Melanoma is 10x costlier than unnecessary biopsy)."""
    def __init__(self, cost_matrix: np.ndarray):
        self.cost_matrix = np.asarray(cost_matrix)

    def compute_total_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if y_true.ndim > 1: y_true = np.argmax(y_true, axis=-1)
        if y_pred.ndim > 1: y_pred = np.argmax(y_pred, axis=-1)
        
        costs = self.cost_matrix[y_true, y_pred]
        return float(np.sum(costs))
