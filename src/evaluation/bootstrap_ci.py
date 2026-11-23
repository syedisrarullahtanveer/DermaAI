"""
Non-parametric bootstrap confidence interval estimator for clinical evaluation metrics.
"""
import numpy as np
from typing import Callable

class BootstrapCIEstimator:
    def __init__(self, n_bootstraps: int = 500, alpha: float = 0.05, seed: int = 42):
        self.n_bootstraps = n_bootstraps
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)

    def estimate(self, metric_fn: Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)
        scores = []
        for _ in range(self.n_bootstraps):
            indices = self.rng.integers(0, n, size=n)
            score = metric_fn(y_true[indices], y_pred[indices])
            scores.append(score)
        
        scores = np.sort(scores)
        low_pct = (self.alpha / 2.0) * 100
        high_pct = (1.0 - self.alpha / 2.0) * 100
        ci_lower = float(np.percentile(scores, low_pct))
        ci_upper = float(np.percentile(scores, high_pct))
        mean_val = float(np.mean(scores))
        return {"mean": mean_val, "ci_lower": ci_lower, "ci_upper": ci_upper}
