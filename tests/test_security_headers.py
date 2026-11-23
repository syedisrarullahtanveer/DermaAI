"""Unit tests for SecurityHeadersMiddleware."""

import unittest
from backend.middleware.security_headers import SecurityHeadersMiddleware


class TestSecurityHeadersMiddleware(unittest.TestCase):
    def test_security_headers_defined(self):
        headers = SecurityHeadersMiddleware.SECURITY_HEADERS
        self.assertIn("X-Content-Type-Options", headers)
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("X-Frame-Options", headers)
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertIn("Strict-Transport-Security", headers)
        self.assertIn("max-age=31536000", headers["Strict-Transport-Security"])


if __name__ == "__main__":
    unittest.main()
