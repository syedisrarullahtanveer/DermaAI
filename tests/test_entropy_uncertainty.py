import unittest
import numpy as np
from src.uncertainty.entropy_uncertainty import PredictiveEntropyQuantifier

class TestPredictiveEntropy(unittest.TestCase):
    def test_zero_entropy_for_deterministic(self):
        quant = PredictiveEntropyQuantifier()
        p = np.array([[1.0, 0.0], [0.0, 1.0]])
        ent = quant.compute_entropy(p)
        self.assertAlmostEqual(ent[0], 0.0, places=4)

    def test_max_entropy_for_uniform(self):
        quant = PredictiveEntropyQuantifier()
        p = np.array([[0.5, 0.5]])
        norm_ent = quant.compute_normalized_entropy(p)
        self.assertAlmostEqual(norm_ent[0], 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
