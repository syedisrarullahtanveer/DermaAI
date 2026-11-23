import unittest
import numpy as np
from src.data.metadata_encoder import MetadataFeatureEncoder

class TestMetadataEncoder(unittest.TestCase):
    def test_vector_length(self):
        encoder = MetadataFeatureEncoder()
        vec = encoder.encode(age=55.0, sex="female", site="torso")
        # 1 age + 3 sex + 7 site = 11 features
        self.assertEqual(len(vec), 11)
        self.assertAlmostEqual(vec[0], 0.55, places=2)

if __name__ == '__main__':
    unittest.main()
