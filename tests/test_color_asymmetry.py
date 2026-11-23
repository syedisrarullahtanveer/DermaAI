import unittest
import numpy as np
from src.preprocessing.color_asymmetry import ColorAsymmetryComputer

class TestColorAsymmetry(unittest.TestCase):
    def test_symmetric_image(self):
        img = np.ones((64, 64, 3), dtype=np.uint8) * 100
        computer = ColorAsymmetryComputer()
        scores = computer.compute_asymmetry(img)
        self.assertEqual(scores["horizontal_asymmetry"], 0.0)
        self.assertEqual(scores["vertical_asymmetry"], 0.0)

if __name__ == '__main__':
    unittest.main()
