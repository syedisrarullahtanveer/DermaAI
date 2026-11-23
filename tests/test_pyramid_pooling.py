import unittest
import tensorflow as tf
from src.models.pyramid_pooling import SpatialPyramidPooling2D

class TestPyramidPooling(unittest.TestCase):
    def test_spp_output_dim(self):
        inputs = tf.random.normal([2, 8, 8, 16])
        spp = SpatialPyramidPooling2D(bin_sizes=[1, 2])
        # Bin 1x1: 1 * 16 = 16; Bin 2x2: 4 * 16 = 64; Total = 80
        out = spp(inputs)
        self.assertEqual(out.shape, (2, 80))

if __name__ == '__main__':
    unittest.main()
