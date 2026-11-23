import unittest
import os

class TestMultiViewGrid(unittest.TestCase):
    def test_file_exists(self):
        path = os.path.join(os.path.dirname(__file__), "..", "frontend", "src", "components", "MultiViewThumbnailGrid.tsx")
        self.assertTrue(os.path.exists(path))
        with open(path, "r", encoding="utf-8") as f:
            self.assertIn("MultiViewThumbnailGrid", f.read())

if __name__ == '__main__':
    unittest.main()
