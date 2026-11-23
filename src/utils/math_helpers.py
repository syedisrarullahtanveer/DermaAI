// Utility mathematical functions used across the project.

import math
from typing import Iterable, List

def sigmoid(x: float) -> float:
    """Return the sigmoid of *x* using a numerically stable formulation."""
    if x >= 0:
        z = math.exp(-x)
        return 1 / (1 + z)
    else:
        z = math.exp(x)
        return z / (1 + z)

def softmax(values: Iterable[float]) -> List[float]:
    """Compute the soft‑max of an iterable of numbers.
    Returns a list of probabilities that sum to 1.0.
    """
    vals = list(values)
    if not vals:
        return []
    max_val = max(vals)
    exp_vals = [math.exp(v - max_val) for v in vals]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]
