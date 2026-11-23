import unittest
from backend.security.api_key_limiter import APIKeyQuotaLimiter

class TestAPIKeyLimiter(unittest.TestCase):
    def test_rate_limiting(self):
        limiter = APIKeyQuotaLimiter()
        # Free tier: 10 rpm
        for _ in range(10):
            allowed, _ = limiter.allow_request("key_123", tier="free")
            self.assertTrue(allowed)
        blocked, msg = limiter.allow_request("key_123", tier="free")
        self.assertFalse(blocked)

if __name__ == '__main__':
    unittest.main()
