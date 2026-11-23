import unittest
import os

class TestClassProbabilityChart(unittest.TestCase):
    def test_component_structure(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "ClassProbabilityBarChart.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
            self.assertIn("ClassProbabilityBarChart", code)
            self.assertIn("probability-chart", code)

if __name__ == '__main__':
    unittest.main()
