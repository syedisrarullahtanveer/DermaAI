import unittest
import tensorflow as tf
from src.models.cbam import CBAM

class TestCBAM(unittest.TestCase):
    def test_cbam_forward_pass(self):
        inputs = tf.random.normal([2, 16, 16, 32])
        cbam = CBAM(in_planes=32)
        out = cbam(inputs)
        self.assertEqual(out.shape, (2, 16, 16, 32))

if __name__ == '__main__':
    unittest.main()
