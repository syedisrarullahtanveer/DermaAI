import unittest
from backend.registry.model_registry import ModelRegistry

class TestModelRegistry(unittest.TestCase):
    def test_register_and_lookup(self):
        reg = ModelRegistry()
        reg.register("2.1.0", {"architecture": "EdgeCNN_v2", "accuracy": 0.932})
        self.assertEqual(reg.list_versions(), ["2.1.0"])
        meta = reg.get_metadata("2.1.0")
        self.assertEqual(meta["architecture"], "EdgeCNN_v2")

if __name__ == '__main__':
    unittest.main()
