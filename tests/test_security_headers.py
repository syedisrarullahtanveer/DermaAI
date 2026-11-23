"""Unit tests for SecurityHeadersMiddleware."""

import unittest
import asyncio
from backend.middleware.security_headers import SecurityHeadersMiddleware


class DummyResponse:
    def __init__(self):
        self.headers = {}


class DummyRequest:
    pass


class TestSecurityHeadersMiddleware(unittest.TestCase):
    def test_adds_owasp_headers(self):
        middleware = SecurityHeadersMiddleware(app=None)
        req = DummyRequest()

        async def dummy_call_next(request):
            return DummyResponse()

        res = asyncio.run(middleware.dispatch(req, dummy_call_next))

        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("default-src 'self'", res.headers.get("Content-Security-Policy", ""))
        self.assertIn("max-age=31536000", res.headers.get("Strict-Transport-Security", ""))


if __name__ == "__main__":
    unittest.main()
