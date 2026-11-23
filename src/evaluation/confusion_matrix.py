"""Multiclass Confusion Matrix and Clinical Diagnostic Metrics Summary."""

from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import confusion_matrix


class ClinicalMetricsSummary:
    """Computes comprehensive per-class clinical metrics: Sensitivity, Specificity, PPV, NPV."""

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        """Initialize clinical metrics analyzer.

        Args:
            class_names: Optional list of diagnostic category names.
        """
        self.class_names = class_names

    def compute_summary(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute confusion matrix and per-class diagnostic statistics.

        Args:
            y_true: Ground truth integer class indices (N,).
            y_pred: Predicted integer class indices (N,).

        Returns:
            Dict[str, Any]: Matrix array, overall accuracy, and per-class metrics dictionary.
        """
        y_true = np.asarray(y_true, dtype=int)
        y_pred = np.asarray(y_pred, dtype=int)

        if len(y_true) != len(y_pred):
            raise ValueError(f"Shape mismatch: len(y_true)={len(y_true)} vs len(y_pred)={len(y_pred)}")

        num_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
        class_names = self.class_names or [f"class_{i}" for i in range(num_classes)]

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        total_samples = len(y_true)
        overall_accuracy = float(np.trace(cm) / total_samples) if total_samples > 0 else 0.0

        per_class: Dict[str, Dict[str, float]] = {}

        for idx in range(num_classes):
            name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            tp = int(cm[idx, idx])
            fn = int(np.sum(cm[idx, :]) - tp)
            fp = int(np.sum(cm[:, idx]) - tp)
            tn = int(total_samples - (tp + fn + fp))

            sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

            per_class[name] = {
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
            }

        return {
            "confusion_matrix": cm.tolist(),
            "overall_accuracy": overall_accuracy,
            "total_samples": total_samples,
            "per_class_metrics": per_class,
        }
