"""
Multi-class Brier score calibration metric.
"""
import numpy as np

class MultiClassBrierScore:
    def compute(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)
        if y_true.ndim == 1:
            num_classes = y_prob.shape[-1]
            y_one_hot = np.zeros((len(y_true), num_classes), dtype=np.float32)
            y_one_hot[np.arange(len(y_true)), y_true] = 1.0
            y_true = y_one_hot
        brier = np.mean(np.sum((y_prob - y_true) ** 2, axis=-1))
        return float(brier)
