import unittest
import numpy as np
from src.preprocessing.letterbox_pad import letterbox_pad

class TestLetterboxPad(unittest.TestCase):
    def test_letterbox_dimensions(self):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        padded, meta = letterbox_pad(img, target_size=(224, 224))
        self.assertEqual(padded.shape, (224, 224, 3))

if __name__ == '__main__':
    unittest.main()
