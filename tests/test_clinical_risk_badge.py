import unittest
import os

class TestClinicalRiskBadge(unittest.TestCase):
    def test_component_file_structure(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "ClinicalRiskBadge.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
            self.assertIn("ClinicalRiskBadge", code)
            self.assertIn("Urgent Biopsy Required", code)

if __name__ == '__main__':
    unittest.main()
