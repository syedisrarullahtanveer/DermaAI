import unittest
import numpy as np
from src.explainability.heatmap_bbox import HeatmapBoundingBoxExtractor

class TestHeatmapBBox(unittest.TestCase):
    def test_bounding_box(self):
        extractor = HeatmapBoundingBoxExtractor(threshold=0.5)
        hm = np.zeros((50, 50), dtype=np.float32)
        hm[10:30, 10:40] = 0.8
        bbox = extractor.extract_bounding_box(hm)
        self.assertIsNotNone(bbox)
        x, y, w, h = bbox
        self.assertEqual((x, y, w, h), (10, 10, 30, 20))

if __name__ == '__main__':
    unittest.main()
