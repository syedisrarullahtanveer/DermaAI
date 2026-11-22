"""Unit Tests: Experiment Runner.

Verifies that ExperimentRunner:
  - Successfully runs a 1-epoch training pass for every model.
  - Produces a results dict with the correct keys.
  - Saves a JSON result file on disk.
  - Saves a training-curve PNG on disk.

Tests run on synthetic random data (no real dataset required) and
complete in reasonable time on CPU.
"""

import json
import sys
from pathlib import Path

# Ensure project root on path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.training.experiment_runner import ExperimentRunner


REQUIRED_RESULT_KEYS = {
    "model_name",
    "epochs_run",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "best_val_loss",
    "best_val_accuracy",
    "duration_seconds",
    "checkpoint_path",
}


def _make_runner() -> ExperimentRunner:
    return ExperimentRunner(
        config_path=str(_PROJECT_ROOT / "configs" / "training_experiments.yaml")
    )


def test_experiment_runner_baseline_cnn():
    """ExperimentRunner completes 1 epoch of baseline_cnn without error."""
    runner = _make_runner()
    result = runner.run("baseline_cnn", epochs=1, batch_size=8)

    missing = REQUIRED_RESULT_KEYS - result.keys()
    assert not missing, f"Result dict missing keys: {missing}"
    assert result["epochs_run"] >= 1
    assert isinstance(result["train_loss"], list)
    assert len(result["train_loss"]) >= 1
    assert 0.0 <= result["best_val_accuracy"] <= 1.0
    print("PASS: test_experiment_runner_baseline_cnn")


def test_experiment_runner_resnet50():
    """ExperimentRunner completes 1 epoch of resnet50 without error."""
    runner = _make_runner()
    result = runner.run("resnet50", epochs=1, batch_size=8)
    assert result["epochs_run"] >= 1
    assert len(result["train_loss"]) >= 1
    print("PASS: test_experiment_runner_resnet50")


def test_experiment_runner_efficientnet_b0():
    """ExperimentRunner completes 1 epoch of efficientnet_b0 without error."""
    runner = _make_runner()
    result = runner.run("efficientnet_b0", epochs=1, batch_size=8)
    assert result["epochs_run"] >= 1
    print("PASS: test_experiment_runner_efficientnet_b0")


def test_experiment_runner_mobilenet_v2():
    """ExperimentRunner completes 1 epoch of mobilenet_v2 without error."""
    runner = _make_runner()
    result = runner.run("mobilenet_v2", epochs=1, batch_size=8)
    assert result["epochs_run"] >= 1
    print("PASS: test_experiment_runner_mobilenet_v2")


def test_result_json_persisted():
    """JSON result file is written to reports/experiments/."""
    runner = _make_runner()
    result = runner.run("baseline_cnn", epochs=1, batch_size=8)

    json_path = Path(result.get("checkpoint_path", "")).parent.parent / "experiments" / "baseline_cnn_results.json"
    # Also check via the standard results_dir
    json_path2 = _PROJECT_ROOT / "reports" / "experiments" / "baseline_cnn_results.json"
    assert json_path2.exists(), f"JSON not found at {json_path2}"
    with open(json_path2) as f:
        data = json.load(f)
    assert data["model_name"] == "baseline_cnn"
    print("PASS: test_result_json_persisted")


def test_training_curve_png_saved():
    """Training curve PNG is saved after a run."""
    runner = _make_runner()
    result = runner.run("baseline_cnn", epochs=1, batch_size=8)
    plot_path = Path(result.get("plot_path", ""))
    assert plot_path.exists(), f"Plot not found: {plot_path}"
    assert plot_path.suffix == ".png"
    print("PASS: test_training_curve_png_saved")


def test_run_all_returns_all_models():
    """run_all returns a result dict for every configured model."""
    runner = _make_runner()
    all_results = runner.run_all(epochs=1, batch_size=8)
    for name in ["baseline_cnn", "resnet50", "efficientnet_b0", "mobilenet_v2"]:
        assert name in all_results, f"Missing result for model: {name}"
    print("PASS: test_run_all_returns_all_models")


if __name__ == "__main__":
    tests = [
        test_experiment_runner_baseline_cnn,
        test_experiment_runner_resnet50,
        test_experiment_runner_efficientnet_b0,
        test_experiment_runner_mobilenet_v2,
        test_result_json_persisted,
        test_training_curve_png_saved,
        test_run_all_returns_all_models,
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
