import unittest
import tensorflow as tf
from src.models.efficientnet_finetune import build_efficientnet_finetune, unfreeze_top_layers

class TestEfficientNetFineTune(unittest.TestCase):
    def test_build_model_shape(self):
        model = build_efficientnet_finetune(num_classes=5, input_shape=(32, 32, 3), variant="B0", freeze_base=True)
        self.assertEqual(model.output_shape, (None, 5))
        # Base layers should be frozen
        base = model.get_layer("efficientnetb0")
        self.assertFalse(base.trainable)

    def test_unfreeze_top_layers(self):
        model = build_efficientnet_finetune(num_classes=5, input_shape=(32, 32, 3), variant="B0", freeze_base=True)
        unfreeze_top_layers(model, num_layers=5)
        base = model.get_layer("efficientnetb0")
        trainable_layers = [l for l in base.layers if l.trainable]
        self.assertGreater(len(trainable_layers), 0)

if __name__ == '__main__':
    unittest.main()
