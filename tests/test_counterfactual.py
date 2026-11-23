import unittest
import numpy as np
from src.explainability.counterfactual import CounterfactualExplainer

class TestCounterfactual(unittest.TestCase):
    def test_counterfactual_bounds(self):
        explainer = CounterfactualExplainer(step_size=0.1, max_iter=5)
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        cf = explainer.generate_counterfactual(img, lambda x: np.ones_like(x), lambda x: np.mean(x) > 0.8)
        self.assertTrue(np.all(cf >= 0.0) and np.all(cf <= 1.0))

if __name__ == '__main__':
    unittest.main()
