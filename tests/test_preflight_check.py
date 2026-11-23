import unittest
import os

class TestPreflightCheck(unittest.TestCase):
    def test_preflight_component(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "PreflightImageQualityCheck.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
            self.assertIn("PreflightImageQualityCheck", code)

if __name__ == '__main__':
    unittest.main()
