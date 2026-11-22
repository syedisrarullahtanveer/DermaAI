"""Training History Visualization for DermaAI.

Provides two public functions:

  plot_training_curves  – saves loss + accuracy curves for a single model run.
  plot_comparison_curves – overlays multiple model histories for benchmarking.

All plots are saved as PNG files; no interactive display is required so the
module works in headless server environments.
"""

import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless backend – must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


_PALETTE = [
    "#2196F3",  # blue
    "#4CAF50",  # green
    "#FF5722",  # deep orange
    "#9C27B0",  # purple
    "#FF9800",  # amber
    "#00BCD4",  # cyan
    "#F44336",  # red
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_training_curves(
    history: Dict[str, List[float]],
    model_name: str,
    output_dir: str,
    *,
    show_validation: bool = True,
) -> str:
    """Plot loss and accuracy curves for a single training run.

    Args:
        history:         Dict with keys such as ``loss``, ``accuracy``,
                         ``val_loss``, ``val_accuracy``.  Matches the
                         ``model.fit().history`` dict returned by Keras.
        model_name:      Human-readable model identifier (used in title / filename).
        output_dir:      Directory where the PNG will be written.
        show_validation: Whether to overlay validation curves when present.

    Returns:
        Absolute path to the saved PNG file.
    """
    output_path = Path(output_dir)
    _ensure_dir(output_path)

    epochs = range(1, len(history.get("loss", [])) + 1)
    has_val = show_validation and "val_loss" in history

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Training History – {model_name}", fontsize=14, fontweight="bold")

    # ── Loss ──────────────────────────────────────────────────────────────────
    ax_loss = axes[0]
    ax_loss.plot(epochs, history.get("loss", []), color=_PALETTE[0],
                 linewidth=2, marker="o", markersize=4, label="Train Loss")
    if has_val:
        ax_loss.plot(epochs, history.get("val_loss", []), color=_PALETTE[2],
                     linewidth=2, marker="s", markersize=4, linestyle="--",
                     label="Val Loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_loss.grid(True, alpha=0.3)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    ax_acc = axes[1]
    acc_key = "accuracy" if "accuracy" in history else "acc"
    val_acc_key = "val_accuracy" if "val_accuracy" in history else "val_acc"
    ax_acc.plot(epochs, history.get(acc_key, []), color=_PALETTE[1],
                linewidth=2, marker="o", markersize=4, label="Train Accuracy")
    if has_val and val_acc_key in history:
        ax_acc.plot(epochs, history.get(val_acc_key, []), color=_PALETTE[3],
                    linewidth=2, marker="s", markersize=4, linestyle="--",
                    label="Val Accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.legend()
    ax_acc.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax_acc.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = output_path / f"{model_name.lower().replace(' ', '_')}_training_curves.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fname.resolve())


def plot_comparison_curves(
    all_histories: Dict[str, Dict[str, List[float]]],
    output_dir: str,
    *,
    metric: str = "accuracy",
) -> Dict[str, str]:
    """Overlay training + validation curves for multiple models.

    Creates two separate PNG files:
      - ``comparison_train_{metric}.png``   – train metric per model
      - ``comparison_val_{metric}.png``     – validation metric per model

    Args:
        all_histories: Mapping of ``{model_name: history_dict}``.
        output_dir:    Directory where PNGs will be written.
        metric:        Which metric to compare (e.g. ``'accuracy'``, ``'loss'``).

    Returns:
        Dict ``{plot_key: absolute_path}`` for train and val plots.
    """
    output_path = Path(output_dir)
    _ensure_dir(output_path)

    val_metric_key = f"val_{metric}"
    train_key = metric if metric != "acc" else "accuracy"
    val_key = val_metric_key if val_metric_key != "val_acc" else "val_accuracy"

    saved = {}

    for phase, key, label in [("train", train_key, "Train"), ("val", val_key, "Val")]:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(
            f"Model Comparison – {label} {metric.capitalize()}",
            fontsize=13, fontweight="bold",
        )
        for idx, (model_name, hist) in enumerate(all_histories.items()):
            if key not in hist:
                continue
            values = hist[key]
            epochs = range(1, len(values) + 1)
            color = _PALETTE[idx % len(_PALETTE)]
            ax.plot(epochs, values, linewidth=2, marker="o", markersize=4,
                    color=color, label=model_name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc="best")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fname = output_path / f"comparison_{phase}_{metric}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved[f"{phase}_{metric}"] = str(fname.resolve())

    return saved


def plot_lr_schedule(
    schedule_fn,
    total_steps: int,
    model_name: str,
    output_dir: str,
) -> str:
    """Visualise how the learning rate evolves across training steps.

    Args:
        schedule_fn:  A callable ``schedule_fn(step) -> lr`` (e.g. a Keras
                      LRSchedule instance).
        total_steps:  Number of steps to simulate.
        model_name:   Used for the plot title and filename.
        output_dir:   Destination directory for the PNG.

    Returns:
        Absolute path to saved PNG.
    """
    import numpy as np

    output_path = Path(output_dir)
    _ensure_dir(output_path)

    steps = np.arange(0, total_steps)
    lrs = [float(schedule_fn(s)) for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, lrs, color=_PALETTE[0], linewidth=2)
    ax.set_title(f"LR Schedule – {model_name}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname = output_path / f"{model_name.lower().replace(' ', '_')}_lr_schedule.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(fname.resolve())
