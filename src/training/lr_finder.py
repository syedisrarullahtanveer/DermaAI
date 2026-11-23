"""
Learning rate range test (LR Finder) for estimating optimal learning rate bounds.
Reference: Smith, "Cyclical Learning Rates for Training Neural Networks", WACV 2017.
"""
import numpy as np

class LRFinder:
    def __init__(self, min_lr: float = 1e-6, max_lr: float = 1e-1, num_steps: int = 100):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.mult = (max_lr / min_lr) ** (1.0 / max_steps if (max_steps := max(1, num_steps - 1)) else 1.0)
        self.lrs = []
        self.losses = []

    def get_lr_for_step(self, step: int) -> float:
        lr = self.min_lr * (self.mult ** step)
        return float(min(self.max_lr, lr))

    def record(self, lr: float, loss: float):
        self.lrs.append(lr)
        self.losses.append(loss)

    def suggest_optimal_lr(self) -> float:
        if len(self.losses) < 5:
            return self.min_lr * 10
        # Optimal lr is where the loss decreases fastest (minimum derivative)
        grads = np.gradient(self.losses)
        best_idx = int(np.argmin(grads))
        return float(self.lrs[best_idx])
