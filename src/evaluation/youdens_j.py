"""
Youden's J statistic index optimizer: J = Sensitivity + Specificity - 1.
Finds the cutoff maximizing informedness.
"""
import numpy as np
from sklearn.metrics import roc_curve

class YoudensJOptimizer:
    def find_best_cutoff(self, y_true: np.ndarray, y_score: np.ndarray) -> dict:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        return {
            "optimal_threshold": float(thresholds[best_idx]),
            "max_j": float(j_scores[best_idx]),
            "sensitivity": float(tpr[best_idx]),
            "specificity": float(1.0 - fpr[best_idx])
        }
