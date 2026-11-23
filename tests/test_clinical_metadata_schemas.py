import unittest
from backend.clinical_metadata import ClinicalQueryMetadata

class TestClinicalSchemas(unittest.TestCase):
    def test_validation(self):
        valid = ClinicalQueryMetadata(patient_age=45, patient_sex="male")
        self.assertEqual(len(valid.validate()), 0)
        
        invalid = ClinicalQueryMetadata(patient_age=150, patient_sex="other")
        self.assertGreater(len(invalid.validate()), 0)

if __name__ == '__main__':
    unittest.main()
