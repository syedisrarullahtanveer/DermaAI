import unittest
import os

class TestInteractiveZoomViewer(unittest.TestCase):
    def test_file_structure(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "InteractiveZoomViewer.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("InteractiveZoomViewer", content)
            self.assertIn("zoomLevel", content)

if __name__ == '__main__':
    unittest.main()
