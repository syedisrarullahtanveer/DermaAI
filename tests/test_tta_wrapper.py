import unittest
import numpy as np
from src.preprocessing.tta_wrapper import TestTimeAugmentor

class TestTTA(unittest.TestCase):
    def test_tta_averaging(self):
        def dummy_pred(batch):
            # Returns [0.8, 0.2]
            return np.array([[0.8, 0.2]])
        tta = TestTimeAugmentor(dummy_pred)
        img = np.ones((32, 32, 3), dtype=np.float32)
        res = tta.predict(img)
        self.assertEqual(res.shape, (2,))
        self.assertAlmostEqual(res[0], 0.8, places=4)

if __name__ == '__main__':
    unittest.main()
