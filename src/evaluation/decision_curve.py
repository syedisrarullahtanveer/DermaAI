"""Clinical Decision Curve Analysis (DCA) and Net Benefit Calculator.

Reference:
    Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: a novel
    method for evaluating prediction models. Medical Decision Making.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class DecisionCurveAnalyzer:
    """Calculates clinical net benefit across decision threshold probabilities."""

    def __init__(self, thresholds: Optional[List[float]] = None) -> None:
        """Initialize DCA analyzer.

        Args:
            thresholds: Array of decision threshold probabilities in (0, 1). Default 1% to 99%.
        """
        if thresholds is None:
            self.thresholds = np.linspace(0.05, 0.95, 19).tolist()
        else:
            self.thresholds = [float(t) for t in thresholds if 0.0 < t < 1.0]

    def compute_net_benefit(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Compute Net Benefit of the model, Treat All strategy, and Treat None strategy.

        Args:
            y_true: Binary ground truth array (1 = Malignant/High risk, 0 = Benign).
            y_prob: Predicted probability for positive class (1D array in [0, 1]).

        Returns:
            Dict[str, Any]: Net benefit series across decision thresholds.
        """
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)
        n = len(y_true)

        if n == 0:
            raise ValueError("y_true cannot be empty")

        prevalence = float(np.mean(y_true))
        curves: List[Dict[str, float]] = []

        for pt in self.thresholds:
            weight = pt / (1.0 - pt)
            y_pred = (y_prob >= pt).astype(int)

            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))

            net_benefit_model = (tp / n) - (fp / n) * weight
            net_benefit_all = prevalence - (1.0 - prevalence) * weight
            net_benefit_none = 0.0

            curves.append({
                "threshold": float(pt),
                "net_benefit_model": float(net_benefit_model),
                "net_benefit_all": float(net_benefit_all),
                "net_benefit_none": float(net_benefit_none),
            })

        return {
            "total_samples": n,
            "prevalence": prevalence,
            "curve_points": curves,
        }
