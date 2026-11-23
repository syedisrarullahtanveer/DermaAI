import unittest
from backend.parsers.dicom_parser import DICOMParser

class TestDICOMParser(unittest.TestCase):
    def test_valid_dicom_header(self):
        parser = DICOMParser()
        raw = b"\x00" * 128 + b"DICM" + b"\x00" * 20
        res = parser.parse_header(raw)
        self.assertTrue(res["valid_dicom"])
        self.assertEqual(res["modality"], "XC")

if __name__ == '__main__':
    unittest.main()
