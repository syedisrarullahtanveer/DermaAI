import unittest
from src.models.efficientnet_v2_scheduler import EfficientNetV2ProgressiveScheduler

class TestProgressiveScheduler(unittest.TestCase):
    def test_progressive_progression(self):
        sched = EfficientNetV2ProgressiveScheduler(min_res=128, max_res=224, total_epochs=10)
        c0 = sched.get_stage_config(0)
        c9 = sched.get_stage_config(9)
        self.assertEqual(c0["image_size"], (128, 128))
        self.assertEqual(c9["image_size"], (224, 224))
        self.assertLess(c0["dropout_rate"], c9["dropout_rate"])

if __name__ == '__main__':
    unittest.main()
