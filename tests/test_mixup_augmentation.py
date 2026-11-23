import unittest
import numpy as np
from src.preprocessing.mixup_augmentation import mixup_pair, mixup_batch

class TestMixUp(unittest.TestCase):
    def test_mixup_pair(self):
        img1 = np.zeros((32, 32, 3), dtype=np.float32)
        lbl1 = np.array([1.0, 0.0], dtype=np.float32)
        img2 = np.ones((32, 32, 3), dtype=np.float32)
        lbl2 = np.array([0.0, 1.0], dtype=np.float32)
        mixed_img, mixed_lbl = mixup_pair(img1, lbl1, img2, lbl2, alpha=1.0)
        self.assertEqual(mixed_img.shape, (32, 32, 3))
        self.assertAlmostEqual(np.sum(mixed_lbl), 1.0, places=4)

    def test_mixup_batch(self):
        imgs = np.ones((6, 16, 16, 3), dtype=np.float32)
        lbls = np.eye(6, dtype=np.float32)
        m_imgs, m_lbls = mixup_batch(imgs, lbls, alpha=0.5, seed=123)
        self.assertEqual(m_imgs.shape, imgs.shape)
        self.assertEqual(m_lbls.shape, lbls.shape)

if __name__ == '__main__':
    unittest.main()
