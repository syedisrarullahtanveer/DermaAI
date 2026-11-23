"""
Dermoscopy dataset integrity and corrupt image verification.
"""
import os
import cv2

class DatasetIntegrityValidator:
    def validate_image_file(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            return False
        img = cv2.imread(file_path)
        return img is not None and img.size > 0
