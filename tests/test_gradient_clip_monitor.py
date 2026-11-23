import unittest
from src.training.gradient_clip_monitor import GradientClipMonitor

class TestGradientClipMonitor(unittest.TestCase):
    def test_monitor_records_norm(self):
        monitor = GradientClipMonitor(warn_threshold=5.0, log_every_n_steps=10)
        for i in range(15):
            monitor.on_train_batch_end(i, logs={"gradient_norm": 2.5 + i * 0.2})
        self.assertEqual(len(monitor.gradient_norm_history), 15)
        self.assertAlmostEqual(monitor.max_gradient_norm, 5.3, places=2)

if __name__ == '__main__':
    unittest.main()
