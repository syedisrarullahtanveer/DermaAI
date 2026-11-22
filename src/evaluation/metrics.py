"""Clinical Multi-Class Diagnostic Metrics for Dermatological Lesion Assessment."""

from typing import Dict, List, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Union[float, Dict[str, float]]]:
    """Compute comprehensive diagnostic metrics: balanced accuracy, precision, recall, F1, and AUC.

    Args:
        y_true: True class indices (1D array) or one-hot encoded matrix.
        y_pred_probs: Predicted probability distribution matrix of shape [N, num_classes].
        class_names: List of class label strings.

    Returns:
        Dict: Structured dictionary of macro, weighted, and per-class clinical metrics.
    """
    if y_true.ndim > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    y_pred_indices = np.argmax(y_pred_probs, axis=1)

    # Core Global Metrics
    overall_acc = float(accuracy_score(y_true_indices, y_pred_indices))
    balanced_acc = float(balanced_accuracy_score(y_true_indices, y_pred_indices))
    macro_precision = float(precision_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true_indices, y_pred_indices, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_indices, y_pred_indices, average="weighted", zero_division=0))

    # Top-2 Accuracy (Crucial for clinical differential diagnosis)
    top_2_correct = 0
    for idx, true_idx in enumerate(y_true_indices):
        top_2_preds = np.argsort(y_pred_probs[idx])[-2:]
        if true_idx in top_2_preds:
            top_2_correct += 1
    top_2_acc = float(top_2_correct / len(y_true_indices))

    # Per-Class Sensitivity (Recall) and Specificity
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(class_names)))
    per_class_metrics = {}

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        per_class_metrics[name] = {
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1_score": f1,
            "support": int(tp + fn),
        }

    # Macro AUC-ROC (One-vs-Rest)
    try:
        macro_auc = float(roc_auc_score(
            tf_one_hot(y_true_indices, len(class_names)),
            y_pred_probs,
            multi_class="ovr",
            average="macro",
        ))
    except Exception:
        macro_auc = 0.0

    return {
        "accuracy": overall_acc,
        "balanced_accuracy": balanced_acc,
        "top_2_accuracy": top_2_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc_roc": macro_auc,
        "per_class": per_class_metrics,
    }


def tf_one_hot(indices: np.ndarray, num_classes: int) -> np.ndarray:
    """Helper to convert integer indices to one-hot binary array."""
    one_hot = np.zeros((len(indices), num_classes), dtype=np.float32)
    one_hot[np.arange(len(indices)), indices] = 1.0
    return one_hot
