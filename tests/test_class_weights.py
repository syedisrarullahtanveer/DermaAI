import unittest
from src.data.class_weights import ClassWeightCalculator

class TestClassWeights(unittest.TestCase):
    def test_weights_inversely_proportional(self):
        calc = ClassWeightCalculator(beta=0.99)
        # 1000 majority vs 10 minority
        weights = calc.compute_weights([1000, 10])
        self.assertGreater(weights[1], weights[0])

if __name__ == '__main__':
    unittest.main()
