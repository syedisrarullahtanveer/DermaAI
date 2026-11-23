import unittest
from backend.security.jwt_auth import JWTAuthHandler

class TestJWTAuth(unittest.TestCase):
    def test_token_creation_and_verification(self):
        auth = JWTAuthHandler("test_secret")
        token = auth.create_token("dr_smith", role="dermatologist")
        payload = auth.verify_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "dr_smith")
        self.assertEqual(payload["role"], "dermatologist")

    def test_tampered_token(self):
        auth = JWTAuthHandler("test_secret")
        token = auth.create_token("dr_smith")
        tampered = token[:-4] + "abcd"
        self.assertIsNone(auth.verify_token(tampered))

if __name__ == '__main__':
    unittest.main()
