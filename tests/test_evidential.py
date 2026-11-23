import unittest
import numpy as np
from src.uncertainty.evidential import EvidentialUncertaintyQuantifier

class TestEvidential(unittest.TestCase):
    def test_vacuity_decrease_with_evidence(self):
        quant = EvidentialUncertaintyQuantifier()
        ev_low = np.array([[0.1, 0.1]])
        ev_high = np.array([[10.0, 10.0]])
        u_low = quant.compute_evidential_metrics(ev_low)["vacuity_uncertainty"]
        u_high = quant.compute_evidential_metrics(ev_high)["vacuity_uncertainty"]
        self.assertGreater(u_low, u_high)

if __name__ == '__main__':
    unittest.main()
