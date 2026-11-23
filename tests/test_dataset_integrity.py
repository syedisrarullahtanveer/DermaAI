import unittest
import tempfile
import os
import cv2
import numpy as np
from src.data.dataset_integrity import DatasetIntegrityValidator

class TestDatasetIntegrity(unittest.TestCase):
    def test_validation(self):
        validator = DatasetIntegrityValidator()
        with tempfile.TemporaryDirectory() as td:
            valid_path = os.path.join(td, "test.jpg")
            img = np.zeros((32, 32, 3), dtype=np.uint8)
            cv2.imwrite(valid_path, img)
            self.assertTrue(validator.validate_image_file(valid_path))
            self.assertFalse(validator.validate_image_file(os.path.join(td, "nonexistent.jpg")))

if __name__ == '__main__':
    unittest.main()
