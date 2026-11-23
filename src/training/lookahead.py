"""
Lookahead optimizer wrapper for stabilized weight trajectory during dermatological model training.
Reference: Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", NeurIPS 2019.
"""
import numpy as np

class LookaheadOptimizer:
    def __init__(self, alpha: float = 0.5, k: int = 5):
        self.alpha = alpha
        self.k = k
        self.step_counter = 0
        self.slow_weights = None

    def initialize(self, current_weights: list):
        self.slow_weights = [np.copy(w) for w in current_weights]
        self.step_counter = 0

    def step(self, fast_weights: list) -> list:
        self.step_counter += 1
        if self.step_counter % self.k == 0:
            for i in range(len(self.slow_weights)):
                self.slow_weights[i] += self.alpha * (fast_weights[i] - self.slow_weights[i])
            return [np.copy(w) for w in self.slow_weights]
        return fast_weights
