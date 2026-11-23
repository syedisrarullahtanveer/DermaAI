"""
Post-hoc temperature scaling probability calibration for neural network logits.
Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""
import numpy as np
from scipy.optimize import minimize

class TemperatureScaler:
    def __init__(self):
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        # Cross entropy objective for T
        def nll(t):
            t_val = max(1e-3, t[0])
            scaled_logits = logits / t_val
            # Softmax
            exp_l = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
            probs = exp_l / np.sum(exp_l, axis=-1, keepdims=True)
            log_p = np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))
            return -np.mean(log_p)

        res = minimize(nll, [1.0], bounds=[(0.05, 10.0)])
        self.temperature = float(res.x[0])
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / self.temperature
        exp_l = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_l / np.sum(exp_l, axis=-1, keepdims=True)
