import unittest
import tensorflow as tf
from src.models.resnet50_finetune import build_resnet50_finetune

class TestResNet50FineTune(unittest.TestCase):
    def test_model_structure(self):
        model = build_resnet50_finetune(num_classes=5, input_shape=(32, 32, 3), freeze_base=True)
        self.assertEqual(model.output_shape, (None, 5))
        base = model.get_layer("resnet50")
        self.assertFalse(base.trainable)

if __name__ == '__main__':
    unittest.main()
