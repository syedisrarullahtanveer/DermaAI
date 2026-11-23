import unittest
from backend.reporting.pdf_report_generator import PDFReportGenerator

class TestPDFReportGenerator(unittest.TestCase):
    def test_report_generation(self):
        gen = PDFReportGenerator()
        html = gen.generate_html_report("CASE-101", {"age": 60, "site": "back"}, [{"name": "Melanoma", "probability": 0.92}], "Urgent Biopsy")
        self.assertIn("CASE-101", html)
        self.assertIn("Melanoma:</strong> 92.0%", html)

if __name__ == '__main__':
    unittest.main()
