import unittest
import os

class TestClinicalReportSummary(unittest.TestCase):
    def test_summary_component(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "ClinicalReportSummary.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
            self.assertIn("ClinicalReportSummary", code)

if __name__ == '__main__':
    unittest.main()
