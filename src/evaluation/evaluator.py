"""Comprehensive Model Evaluator for Generating Reports and Visual Diagnostics."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.evaluation.metrics import compute_classification_metrics
from src.utils import setup_logger

logger = setup_logger("DermaAI.Evaluator")


class ModelEvaluator:
    """Generates evaluation matrices, confusion plots, and serializable performance reports."""

    def __init__(self, class_names: List[str]):
        self.class_names = class_names

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred_probs: np.ndarray,
        output_dir: Optional[str] = "reports",
        report_prefix: str = "model_evaluation",
    ) -> Dict:
        """Run full evaluation, calculate metrics, render confusion matrix, and save reports."""
        metrics = compute_classification_metrics(y_true, y_pred_probs, self.class_names)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save metrics JSON
            json_file = out_path / f"{report_prefix}_metrics.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Evaluation metrics report saved to: {json_file}")

            # Plot Confusion Matrix
            if y_true.ndim > 1:
                y_true_indices = np.argmax(y_true, axis=1)
            else:
                y_true_indices = y_true
            y_pred_indices = np.argmax(y_pred_probs, axis=1)

            cm = confusion_matrix(y_true_indices, y_pred_indices, labels=range(len(self.class_names)))
            self._plot_confusion_matrix(cm, out_path / f"{report_prefix}_confusion_matrix.png")

        return metrics

    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: Path) -> None:
        """Generate and save publication-quality confusion matrix heatmap."""
        plt.figure(figsize=(9, 7))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar=True,
        )
        plt.title("DermaAI: Multi-Class Confusion Matrix", fontsize=14, pad=15)
        plt.xlabel("Predicted Diagnosis", fontsize=12)
        plt.ylabel("Ground Truth Diagnosis", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {save_path}")
