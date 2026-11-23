import unittest
import tensorflow as tf
from src.training.gradient_accumulation import GradientAccumulator

class TestGradientAccumulation(unittest.TestCase):
    def test_accumulation_and_reset(self):
        acc = GradientAccumulator(steps=2)
        g1 = [tf.constant([1.0, 2.0])]
        acc.accumulate(g1)
        self.assertFalse(acc.should_apply())
        
        g2 = [tf.constant([3.0, 4.0])]
        acc.accumulate(g2)
        self.assertTrue(acc.should_apply())
        
        scaled = acc.get_and_reset()
        self.assertEqual(float(scaled[0][0]), 2.0)
        self.assertEqual(acc.current_step, 0)

if __name__ == '__main__':
    unittest.main()
