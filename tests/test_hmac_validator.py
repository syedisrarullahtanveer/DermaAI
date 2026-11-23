import unittest
from backend.security.hmac_validator import HMACSignatureValidator

class TestHMAC(unittest.TestCase):
    def test_signature_validation(self):
        val = HMACSignatureValidator("derma_secret_key_2026")
        data = b"image_payload_bytes_sample"
        sig = val.generate_signature(data)
        self.assertTrue(val.verify_signature(data, sig))
        self.assertFalse(val.verify_signature(b"tampered", sig))

if __name__ == '__main__':
    unittest.main()
