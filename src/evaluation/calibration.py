"""Clinical Model Calibration Metrics and Reliability Evaluation.

Calculates Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
and bin-wise accuracy/confidence statistics for clinical safety audits.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On Calibration of Modern Neural Networks. In ICML 2017.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


class ModelCalibrationEvaluator:
    """Evaluate confidence calibration for dermatological classification models."""

    def __init__(self, num_bins: int = 10) -> None:
        """Initialize calibration evaluator.

        Args:
            num_bins: Number of equal-width confidence intervals in [0, 1].
        """
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        self.num_bins = num_bins

    def compute_ece(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, Any]:
        """Compute Expected Calibration Error (ECE) and bin statistics.

        Args:
            y_true: Ground truth labels (1D integer class array or 2D one-hot array).
            y_prob: Predicted class probability distributions of shape (N, C).

        Returns:
            Dict[str, Any]: Calibration statistics including ece, mce, and bin details.
        """
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        if y_prob.ndim != 2:
            raise ValueError(f"y_prob must be 2D array of shape (N, C), got {y_prob.shape}")

        if len(y_true) != len(y_prob):
            raise ValueError(f"Length mismatch: len(y_true)={len(y_true)} vs len(y_prob)={len(y_prob)}")

        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(np.float64)

        bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)
        total_samples = len(y_true)

        ece = 0.0
        mce = 0.0
        bin_details: List[Dict[str, float]] = []

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            if i == self.num_bins - 1:
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            prop_in_bin = np.mean(in_bin)
            bin_count = int(np.sum(in_bin))

            if bin_count > 0:
                accuracy_in_bin = float(np.mean(accuracies[in_bin]))
                confidence_in_bin = float(np.mean(confidences[in_bin]))
                gap = abs(accuracy_in_bin - confidence_in_bin)

                ece += prop_in_bin * gap
                mce = max(mce, gap)
            else:
                accuracy_in_bin = 0.0
                confidence_in_bin = 0.0
                gap = 0.0

            bin_details.append({
                "bin_index": i,
                "bin_lower": float(bin_lower),
                "bin_upper": float(bin_upper),
                "sample_count": bin_count,
                "accuracy": accuracy_in_bin,
                "confidence": confidence_in_bin,
                "calibration_gap": float(gap),
            })

        return {
            "expected_calibration_error": float(ece),
            "maximum_calibration_error": float(mce),
            "total_samples": total_samples,
            "bins": bin_details,
        }
