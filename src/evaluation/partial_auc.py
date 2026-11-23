"""
Clinical partial AUC (pAUC) for screening: restricts ROC integration to high-specificity region (e.g. Spec >= 80%).
"""
import numpy as np
from sklearn.metrics import roc_curve

class PartialAUCEvaluator:
    def __init__(self, min_specificity: float = 0.80):
        self.max_fpr = 1.0 - min_specificity

    def compute_pauc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        # Filter points within max_fpr
        mask = fpr <= self.max_fpr
        sub_fpr = np.concatenate([fpr[mask], [self.max_fpr]])
        sub_tpr = np.concatenate([tpr[mask], [np.interp(self.max_fpr, fpr, tpr)]])
        pauc = np.trapz(sub_tpr, sub_fpr)
        # Standardize pAUC to [0, 1]
        norm_pauc = pauc / self.max_fpr
        return float(norm_pauc)
