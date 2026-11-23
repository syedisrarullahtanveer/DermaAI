import unittest
import os

class TestABCDScoreCard(unittest.TestCase):
    def test_component(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "ABCDScoreCard.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
            self.assertIn("ABCDScoreCard", code)
            self.assertIn("Total Dermatoscopy Value", code)

if __name__ == '__main__':
    unittest.main()
