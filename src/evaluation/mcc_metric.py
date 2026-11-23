"""
Matthews Correlation Coefficient (MCC) evaluator for highly imbalanced dermoscopy classification.
"""
import numpy as np
from sklearn.metrics import matthews_corrcoef

class MulticlassMCC:
    def compute(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_t = np.asarray(y_true)
        y_p = np.asarray(y_pred)
        if y_t.ndim > 1:
            y_t = np.argmax(y_t, axis=-1)
        if y_p.ndim > 1:
            y_p = np.argmax(y_p, axis=-1)
        return float(matthews_corrcoef(y_t, y_p))
