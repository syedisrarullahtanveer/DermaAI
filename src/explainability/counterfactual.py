"""
Counterfactual explanation generator via gradient-guided pixel perturbation.
"""
import numpy as np

class CounterfactualExplainer:
    def __init__(self, step_size: float = 0.05, max_iter: int = 20):
        self.step_size = step_size
        self.max_iter = max_iter

    def generate_counterfactual(self, image: np.ndarray, grad_fn, target_fn) -> np.ndarray:
        cf = image.copy().astype(np.float32)
        for _ in range(self.max_iter):
            if target_fn(cf):
                break
            g = grad_fn(cf)
            cf += self.step_size * np.sign(g)
            cf = np.clip(cf, 0.0, 1.0)
        return cf
