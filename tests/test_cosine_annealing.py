"""Unit tests for CosineAnnealingWarmRestarts learning rate schedule."""

import unittest
import tensorflow as tf
from src.training.cosine_annealing import CosineAnnealingWarmRestarts


class TestCosineAnnealingWarmRestarts(unittest.TestCase):
    def test_initial_and_final_learning_rates(self):
        schedule = CosineAnnealingWarmRestarts(
            initial_learning_rate=0.01,
            min_learning_rate=0.001,
            first_cycle_steps=100,
            cycle_mult=1.0,
        )

        # Step 0: should be initial learning rate
        lr_0 = schedule(0).numpy()
        self.assertAlmostEqual(lr_0, 0.01, places=5)

        # Step 100: end of cycle 1, should be min learning rate
        lr_100 = schedule(100).numpy()
        self.assertAlmostEqual(lr_100, 0.001, places=5)

    def test_warm_restart_resets_learning_rate(self):
        schedule = CosineAnnealingWarmRestarts(
            initial_learning_rate=0.01,
            min_learning_rate=0.001,
            first_cycle_steps=100,
            cycle_mult=1.0,
        )
        # Step 101: start of new cycle
        lr_101 = schedule(101).numpy()
        # Should be back near 0.01
        self.assertGreater(lr_101, 0.009)

    def test_serialization(self):
        schedule = CosineAnnealingWarmRestarts(
            initial_learning_rate=0.005,
            min_learning_rate=0.0001,
            first_cycle_steps=50,
            cycle_mult=2.0,
        )
        config = schedule.get_config()
        self.assertEqual(config["first_cycle_steps"], 50)
        self.assertEqual(config["cycle_mult"], 2.0)


if __name__ == "__main__":
    unittest.main()
