import unittest
from src.utils.math_helpers import sigmoid, softmax

class TestMathHelpers(unittest.TestCase):
    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(10), 1.0, places=5)
        self.assertAlmostEqual(sigmoid(-10), 0.0, places=5)

    def test_softmax(self):
        vals = [1.0, 2.0, 3.0]
        probs = softmax(vals)
        self.assertAlmostEqual(sum(probs), 1.0)
        # Check ordering
        self.assertTrue(probs[2] > probs[1] > probs[0])

if __name__ == "__main__":
    unittest.main()
