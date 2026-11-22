"""Unit Tests: History Visualizer.

Tests that plot_training_curves and plot_comparison_curves:
  - Produce PNG files on disk.
  - Handle missing validation data gracefully.
  - Handle multiple models in comparison mode.
  - Return correct file paths.
"""

import sys
import tempfile
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.history_visualizer import (
    plot_comparison_curves,
    plot_training_curves,
    plot_lr_schedule,
)


SAMPLE_HISTORY = {
    "loss": [1.9, 1.7, 1.5, 1.3, 1.1],
    "accuracy": [0.14, 0.20, 0.30, 0.38, 0.45],
    "val_loss": [2.1, 2.0, 1.8, 1.6, 1.5],
    "val_accuracy": [0.10, 0.15, 0.22, 0.30, 0.35],
}


def test_plot_training_curves_saves_png():
    """plot_training_curves returns a path to an existing PNG."""
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_training_curves(SAMPLE_HISTORY, "baseline_cnn", tmp)
        assert Path(path).exists(), f"PNG not created: {path}"
        assert path.endswith(".png")
    print("PASS: test_plot_training_curves_saves_png")


def test_plot_training_curves_no_validation():
    """plot_training_curves works when no validation keys are present."""
    history_no_val = {
        "loss": [1.9, 1.7, 1.5],
        "accuracy": [0.14, 0.20, 0.30],
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_training_curves(history_no_val, "test_model", tmp)
        assert Path(path).exists()
    print("PASS: test_plot_training_curves_no_validation")


def test_plot_training_curves_filename_uses_model_name():
    """PNG filename includes (lowercased) model name."""
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_training_curves(SAMPLE_HISTORY, "ResNet50", tmp)
        fname = Path(path).name
        assert "resnet50" in fname.lower(), f"Model name not in filename: {fname}"
    print("PASS: test_plot_training_curves_filename_uses_model_name")


def test_plot_comparison_curves_creates_two_files():
    """plot_comparison_curves creates train and val PNG files."""
    all_histories = {
        "baseline_cnn": SAMPLE_HISTORY,
        "resnet50": {
            "loss": [1.8, 1.6, 1.4, 1.2, 1.0],
            "accuracy": [0.16, 0.22, 0.32, 0.40, 0.48],
            "val_loss": [2.0, 1.9, 1.7, 1.5, 1.4],
            "val_accuracy": [0.12, 0.18, 0.25, 0.32, 0.38],
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        saved = plot_comparison_curves(all_histories, tmp, metric="accuracy")
        assert "train_accuracy" in saved
        assert "val_accuracy" in saved
        for key, path in saved.items():
            assert Path(path).exists(), f"File missing: {path}"
    print("PASS: test_plot_comparison_curves_creates_two_files")


def test_plot_comparison_curves_loss_metric():
    """plot_comparison_curves works for metric='loss'."""
    all_histories = {
        "model_a": SAMPLE_HISTORY,
        "model_b": SAMPLE_HISTORY,
    }
    with tempfile.TemporaryDirectory() as tmp:
        saved = plot_comparison_curves(all_histories, tmp, metric="loss")
        assert "train_loss" in saved
        assert "val_loss" in saved
    print("PASS: test_plot_comparison_curves_loss_metric")


def test_plot_lr_schedule_saves_png():
    """plot_lr_schedule saves a PNG file."""
    from src.training.lr_scheduler import CosineAnnealingSchedule

    sched = CosineAnnealingSchedule(initial_lr=1e-3, t_max=50, eta_min=1e-6, steps_per_epoch=1)
    with tempfile.TemporaryDirectory() as tmp:
        path = plot_lr_schedule(sched, total_steps=50, model_name="cosine_test", output_dir=tmp)
        assert Path(path).exists(), f"LR schedule PNG not created: {path}"
    print("PASS: test_plot_lr_schedule_saves_png")


if __name__ == "__main__":
    tests = [
        test_plot_training_curves_saves_png,
        test_plot_training_curves_no_validation,
        test_plot_training_curves_filename_uses_model_name,
        test_plot_comparison_curves_creates_two_files,
        test_plot_comparison_curves_loss_metric,
        test_plot_lr_schedule_saves_png,
    ]

    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed:
        sys.exit(1)
