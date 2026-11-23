import unittest
from src.data.leakage_detector import PatientLeakageDetector

class TestLeakageDetector(unittest.TestCase):
    def test_detects_overlap(self):
        detector = PatientLeakageDetector()
        train_p = ["P001", "P002", "P003"]
        val_p = ["P003", "P004"]
        leakage = detector.detect_leakage(train_p, val_p)
        self.assertEqual(leakage, {"P003"})
        self.assertFalse(detector.is_clean(train_p, val_p))

    def test_clean_split(self):
        detector = PatientLeakageDetector()
        self.assertTrue(detector.is_clean(["P1", "P2"], ["P3", "P4"]))

if __name__ == '__main__':
    unittest.main()
