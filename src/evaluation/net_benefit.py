"""
Net Benefit decision curve analysis for clinical threshold utility.
Reference: Vickers & Elkin, "Decision Curve Analysis", Med Decis Making 2006.
"""
import numpy as np

class NetBenefitAnalyzer:
    def compute_net_benefit(self, y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        n = len(y_true)
        preds = (y_prob >= threshold).astype(int)
        tp = np.sum((y_true == 1) & (preds == 1))
        fp = np.sum((y_true == 0) & (preds == 1))
        weight = threshold / (1.0 - threshold)
        net_benefit = (tp / n) - (fp / n) * weight
        return float(net_benefit)
