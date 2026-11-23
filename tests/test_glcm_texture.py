import unittest
import numpy as np
from src.preprocessing.glcm_texture import GLCMTextureExtractor

class TestGLCMTexture(unittest.TestCase):
    def test_uniform_image(self):
        extractor = GLCMTextureExtractor()
        img = np.ones((32, 32), dtype=np.uint8) * 100
        contrast = extractor.compute_contrast(img)
        homogeneity = extractor.compute_homogeneity(img)
        self.assertEqual(contrast, 0.0)
        self.assertEqual(homogeneity, 1.0)

if __name__ == '__main__':
    unittest.main()
