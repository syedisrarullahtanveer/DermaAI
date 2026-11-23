import unittest
import numpy as np
from src.data.feature_oversampler import FeatureSpaceOversampler

class TestFeatureOversampler(unittest.TestCase):
    def test_resample_count(self):
        sampler = FeatureSpaceOversampler(seed=42)
        feats = np.random.randn(5, 10)
        res = sampler.resample_class(feats, target_count=12)
        self.assertEqual(len(res), 12)

if __name__ == '__main__':
    unittest.main()
