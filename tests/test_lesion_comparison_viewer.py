"""Structural validation test for LesionComparisonViewer React component."""

import os
import unittest


class TestLesionComparisonViewerComponent(unittest.TestCase):
    def setUp(self):
        self.file_path = os.path.join(
            os.path.dirname(__file__), "..", "frontend", "src", "components", "LesionComparisonViewer.tsx"
        )

    def test_component_file_exists(self):
        self.assertTrue(os.path.isfile(self.file_path))

    def test_component_interface_and_features(self):
        with open(self.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check required TypeScript interfaces and props
        self.assertIn("interface LesionComparisonViewerProps", content)
        self.assertIn("originalImage", content)
        self.assertIn("gradcamOverlay", content)
        # Check slider and side-by-side view modes
        self.assertIn("sliderPosition", content)
        self.assertIn("side-by-side", content)
        self.assertIn("Split Slider", content)


if __name__ == "__main__":
    unittest.main()
