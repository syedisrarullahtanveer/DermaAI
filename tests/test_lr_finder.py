import unittest
from src.training.lr_finder import LRFinder

class TestLRFinder(unittest.TestCase):
    def test_schedule_and_suggestion(self):
        finder = LRFinder(min_lr=1e-4, max_lr=1e-2, num_steps=10)
        lr0 = finder.get_lr_for_step(0)
        lr9 = finder.get_lr_for_step(9)
        self.assertAlmostEqual(lr0, 1e-4, places=6)
        self.assertAlmostEqual(lr9, 1e-2, places=4)
        
        # Feed declining then rising curve
        for s in range(10):
            lr = finder.get_lr_for_step(s)
            loss = (s - 5) ** 2
            finder.record(lr, loss)
        suggested = finder.suggest_optimal_lr()
        self.assertGreater(suggested, 0)

if __name__ == '__main__':
    unittest.main()
