import unittest
import numpy as np
from backend.ensemble.ensemble_coordinator import EnsembleCoordinator

class TestEnsembleCoordinator(unittest.TestCase):
    def test_weighted_combination(self):
        coord = EnsembleCoordinator({"m1": 0.5, "m2": 0.5})
        out1 = np.array([1.0, 0.0])
        out2 = np.array([0.0, 1.0])
        comb = coord.combine_predictions({"m1": out1, "m2": out2})
        self.assertAlmostEqual(comb[0], 0.5)
        self.assertAlmostEqual(comb[1], 0.5)

if __name__ == '__main__':
    unittest.main()
