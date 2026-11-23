import unittest
import time
from backend.cache.inference_cache import InferenceCache

class TestInferenceCache(unittest.TestCase):
    def test_lru_and_ttl(self):
        cache = InferenceCache(max_size=2, ttl_seconds=1)
        cache.set("a", 100)
        cache.set("b", 200)
        self.assertEqual(cache.get("a"), 100)
        cache.set("c", 300) # evicts b
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 300)

if __name__ == '__main__':
    unittest.main()
