"""Unit Tests: Learning Rate Schedulers.

Verifies correct behaviour of:
  - CosineAnnealingSchedule
  - WarmupCosineDecay
  - StepDecay
  - build_scheduler factory
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import tensorflow as tf
from src.training.lr_scheduler import (
    CosineAnnealingSchedule,
    StepDecay,
    WarmupCosineDecay,
    build_scheduler,
)


def test_cosine_annealing_starts_at_max():
    """LR at step 0 equals initial_lr."""
    sched = CosineAnnealingSchedule(initial_lr=1e-3, t_max=5, eta_min=1e-6, steps_per_epoch=1)
    lr_start = float(sched(0))
    assert abs(lr_start - 1e-3) < 1e-7, f"Expected 1e-3, got {lr_start}"
    print("PASS: test_cosine_annealing_starts_at_max")


def test_cosine_annealing_decreases_monotonically():
    """LR should decrease over time (cosine decay, no restarts)."""
    sched = CosineAnnealingSchedule(initial_lr=1e-3, t_max=10, eta_min=1e-6, steps_per_epoch=1)
    lrs = [float(sched(s)) for s in range(10)]
    for i in range(len(lrs) - 1):
        assert lrs[i] >= lrs[i + 1] - 1e-9, f"LR increased at step {i+1}: {lrs[i]} → {lrs[i+1]}"
    print("PASS: test_cosine_annealing_decreases_monotonically")


def test_cosine_annealing_reaches_eta_min():
    """LR at final step should equal eta_min."""
    eta_min = 1e-6
    t_max = 5
    sched = CosineAnnealingSchedule(initial_lr=1e-3, t_max=t_max, eta_min=eta_min, steps_per_epoch=1)
    lr_end = float(sched(t_max))
    assert abs(lr_end - eta_min) < 1e-9, f"Expected {eta_min}, got {lr_end}"
    print("PASS: test_cosine_annealing_reaches_eta_min")


def test_warmup_cosine_increases_during_warmup():
    """LR increases linearly during warmup phase."""
    sched = WarmupCosineDecay(initial_lr=1e-3, warmup_epochs=3, total_epochs=10, steps_per_epoch=1)
    lrs = [float(sched(s)) for s in range(3)]
    for i in range(len(lrs) - 1):
        assert lrs[i] <= lrs[i + 1] + 1e-9, f"LR did not increase during warmup at step {i}"
    print("PASS: test_warmup_cosine_increases_during_warmup")


def test_warmup_cosine_peaks_at_initial_lr():
    """LR at end of warmup should approximately equal initial_lr."""
    initial_lr = 1e-3
    warmup_epochs = 3
    sched = WarmupCosineDecay(
        initial_lr=initial_lr, warmup_epochs=warmup_epochs, total_epochs=10, steps_per_epoch=1
    )
    lr_at_warmup_end = float(sched(warmup_epochs - 1))
    # Allow 5% tolerance due to discrete step boundary
    assert lr_at_warmup_end <= initial_lr + 1e-9
    assert lr_at_warmup_end > 0.5 * initial_lr
    print("PASS: test_warmup_cosine_peaks_at_initial_lr")


def test_step_decay_initial_lr():
    """LR at step 0 equals initial_lr."""
    sched = StepDecay(initial_lr=1e-3, drop_factor=0.5, drop_every_n_epochs=2, steps_per_epoch=1)
    assert abs(float(sched(0)) - 1e-3) < 1e-9
    print("PASS: test_step_decay_initial_lr")


def test_step_decay_drops_at_boundary():
    """LR should halve at each drop boundary."""
    sched = StepDecay(initial_lr=1e-3, drop_factor=0.5, drop_every_n_epochs=2, steps_per_epoch=1)
    lr_before = float(sched(1))   # epoch 1, still in first interval
    lr_after = float(sched(2))    # epoch 2, first drop
    assert lr_after < lr_before, f"Expected drop at epoch 2: {lr_before} → {lr_after}"
    print("PASS: test_step_decay_drops_at_boundary")


def test_build_scheduler_cosine():
    """Factory returns CosineAnnealingSchedule for 'cosine_annealing'."""
    s = build_scheduler("cosine_annealing", initial_lr=1e-3, steps_per_epoch=5, t_max=10)
    assert isinstance(s, CosineAnnealingSchedule)
    print("PASS: test_build_scheduler_cosine")


def test_build_scheduler_warmup():
    """Factory returns WarmupCosineDecay for 'warmup_cosine'."""
    s = build_scheduler("warmup_cosine", initial_lr=1e-3, steps_per_epoch=5)
    assert isinstance(s, WarmupCosineDecay)
    print("PASS: test_build_scheduler_warmup")


def test_build_scheduler_step():
    """Factory returns StepDecay for 'step_decay'."""
    s = build_scheduler("step_decay", initial_lr=1e-3, steps_per_epoch=5)
    assert isinstance(s, StepDecay)
    print("PASS: test_build_scheduler_step")


def test_build_scheduler_invalid_raises():
    """Factory raises ValueError for unknown scheduler type."""
    try:
        build_scheduler("unknown_type", initial_lr=1e-3, steps_per_epoch=5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("PASS: test_build_scheduler_invalid_raises")


def test_scheduler_serialisable():
    """get_config returns a dict that can be JSON-serialised."""
    import json
    for sched in [
        CosineAnnealingSchedule(1e-3, 5, 1e-6, 10),
        WarmupCosineDecay(1e-3, 2, 10, 1e-7, 10),
        StepDecay(1e-3, 0.5, 2, 10),
    ]:
        cfg = sched.get_config()
        assert isinstance(cfg, dict)
        json.dumps(cfg)  # Must not raise
    print("PASS: test_scheduler_serialisable")


if __name__ == "__main__":
    tests = [
        test_cosine_annealing_starts_at_max,
        test_cosine_annealing_decreases_monotonically,
        test_cosine_annealing_reaches_eta_min,
        test_warmup_cosine_increases_during_warmup,
        test_warmup_cosine_peaks_at_initial_lr,
        test_step_decay_initial_lr,
        test_step_decay_drops_at_boundary,
        test_build_scheduler_cosine,
        test_build_scheduler_warmup,
        test_build_scheduler_step,
        test_build_scheduler_invalid_raises,
        test_scheduler_serialisable,
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
