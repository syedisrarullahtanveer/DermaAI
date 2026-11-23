"""Multiclass Receiver Operating Characteristic (ROC) and Area Under the Curve (AUC) Metrics.

Computes One-vs-Rest (OvR) per-class AUC, unweighted macro-average AUC,
and prevalence-weighted AUC for imbalanced dermoscopic classification.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


class MulticlassROCEvaluator:
    """Compute per-class and aggregated Area Under the ROC Curve metrics."""

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """Initialize ROC evaluator.

        Args:
            class_names: Optional human-readable labels for each class index.
        """
        self.class_names = class_names

    def evaluate(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Any]:
        """Compute OvR ROC-AUC metrics for multiclass predictions.

        Args:
            y_true: Ground truth labels (1D integer class array or 2D one-hot array).
            y_score: Predicted class probability distributions of shape (N, C).

        Returns:
            Dict[str, Any]: Comprehensive AUC scores and per-class performance.
        """
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        if y_true.ndim == 2:
            y_true_indices = np.argmax(y_true, axis=1)
            y_true_onehot = y_true
        else:
            y_true_indices = y_true
            num_classes = y_score.shape[1]
            y_true_onehot = np.eye(num_classes)[y_true_indices]

        num_classes = y_score.shape[1]
        class_names = self.class_names or [f"class_{i}" for i in range(num_classes)]

        macro_auc = float(roc_auc_score(y_true_onehot, y_score, multi_class="ovr", average="macro"))
        weighted_auc = float(roc_auc_score(y_true_onehot, y_score, multi_class="ovr", average="weighted"))

        per_class_auc: Dict[str, float] = {}
        for idx in range(num_classes):
            name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            # Check if class is present in y_true
            if np.sum(y_true_onehot[:, idx]) > 0 and np.sum(y_true_onehot[:, idx]) < len(y_true):
                score = float(roc_auc_score(y_true_onehot[:, idx], y_score[:, idx]))
                per_class_auc[name] = score
            else:
                per_class_auc[name] = 0.0

        return {
            "macro_auc": macro_auc,
            "weighted_auc": weighted_auc,
            "per_class_auc": per_class_auc,
            "num_classes": num_classes,
        }
