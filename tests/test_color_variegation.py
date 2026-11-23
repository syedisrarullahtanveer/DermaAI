import unittest
import numpy as np
from src.preprocessing.color_variegation import ColorVariegationCalculator

class TestColorVariegation(unittest.TestCase):
    def test_variegation_scoring(self):
        calc = ColorVariegationCalculator()
        img = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.uint8)
        res = calc.calculate_color_score(img, mask)
        self.assertIn("variegation_index", res)
        self.assertGreater(res["num_colors"], 0)

if __name__ == '__main__':
    unittest.main()
