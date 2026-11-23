"""
Adaptive Expected Calibration Error (AdaECE) using equal-frequency quantile binning.
Reference: Ding et al., "Revisiting the Evaluation of Model Calibration", arXiv 2020.
"""
import numpy as np

class AdaptiveECE:
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def compute(self, confidences: np.ndarray, accuracies: np.ndarray) -> float:
        n = len(confidences)
        if n == 0:
            return 0.0
        
        # Sort by confidence
        order = np.argsort(confidences)
        conf_sorted = confidences[order]
        acc_sorted = accuracies[order]
        
        bins = np.array_split(np.arange(n), self.num_bins)
        ece = 0.0
        for b in bins:
            if len(b) == 0: continue
            bin_conf = np.mean(conf_sorted[b])
            bin_acc = np.mean(acc_sorted[b])
            ece += (len(b) / n) * np.abs(bin_conf - bin_acc)
            
        return float(ece)
