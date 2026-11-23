import unittest
import os

class TestUncertaintyGauge(unittest.TestCase):
    def test_gauge(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "UncertaintyGauge.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            self.assertIn("UncertaintyGauge", f.read())

if __name__ == '__main__':
    unittest.main()
