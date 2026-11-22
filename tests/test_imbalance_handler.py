"""Unit Tests for Class Imbalance Handler."""

import numpy as np
from src.preprocessing.imbalance_handler import ClassImbalanceHandler


def test_compute_balanced_weights():
    """Verify balanced heuristic weight computation."""
    # 3 classes: nv (1000), mel (100), df (10)
    counts = {"nv": 1000, "mel": 100, "df": 10}
    weights = ClassImbalanceHandler.compute_balanced_weights(counts)

    # Minority class 'df' must have the highest weight
    assert weights["df"] > weights["mel"] > weights["nv"]
    # Check exact formula: total = 1110, k = 3 -> w_nv = 1110 / (3 * 1000) = 0.37
    assert round(weights["nv"], 2) == 0.37
    assert round(weights["df"], 2) == 37.0


def test_effective_number_weights():
    """Verify Cui et al. effective number weights sum to number of classes."""
    counts = {"nv": 6705, "mel": 1113, "bcc": 514, "df": 115}
    weights = ClassImbalanceHandler.compute_effective_number_weights(counts, beta=0.999)

    assert len(weights) == 4
    # Sum of normalized weights should equal K (4)
    np.testing.assert_allclose(sum(weights.values()), 4.0, rtol=1e-5)
    assert weights["df"] > weights["nv"]


def test_augmentation_multipliers():
    """Verify minority classes receive higher augmentation multipliers."""
    counts = {"nv": 1000, "mel": 200, "df": 20}
    multipliers = ClassImbalanceHandler.compute_augmentation_multipliers(
        counts, target_fraction_of_max=0.5, max_multiplier=10
    )

    # nv is already >= 500, so multiplier = 1
    assert multipliers["nv"] == 1
    # mel is 200, target 500 -> multiplier = ceil(500/200) = 3
    assert multipliers["mel"] == 3
    # df is 20, target 500 -> ceil(500/20) = 25, capped at max_multiplier = 10
    assert multipliers["df"] == 10
