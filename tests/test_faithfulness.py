import unittest
import numpy as np
from src.explainability.faithfulness import FaithfulnessEvaluator

class TestFaithfulness(unittest.TestCase):
    def test_deletion_evaluation(self):
        evaluator = FaithfulnessEvaluator()
        img = np.ones((10, 10, 3), dtype=np.float32)
        sal = np.random.rand(10, 10)
        score = evaluator.compute_deletion_score(lambda x: float(np.mean(x)), img, sal, steps=2)
        self.assertGreaterEqual(score, 0.0)

if __name__ == '__main__':
    unittest.main()
