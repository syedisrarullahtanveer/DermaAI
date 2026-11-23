"""Unit tests for InMemoryRateLimiter."""

import unittest
import time
from backend.middleware.rate_limiter import InMemoryRateLimiter


class TestInMemoryRateLimiter(unittest.TestCase):
    def test_allows_up_to_max_requests(self):
        limiter = InMemoryRateLimiter(max_requests=3, window_seconds=10.0)
        client = "192.168.1.100"

        # Request 1
        allowed, remaining, _ = limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 2)

        # Request 2
        allowed, remaining, _ = limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 1)

        # Request 3
        allowed, remaining, _ = limiter.is_allowed(client)
        self.assertTrue(allowed)
        self.assertEqual(remaining, 0)

        # Request 4: limit exceeded
        allowed, remaining, retry_after = limiter.is_allowed(client)
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(retry_after, 0.0)

    def test_isolated_per_client(self):
        limiter = InMemoryRateLimiter(max_requests=1, window_seconds=5.0)
        client_a = "10.0.0.1"
        client_b = "10.0.0.2"

        allowed_a, _, _ = limiter.is_allowed(client_a)
        allowed_b, _, _ = limiter.is_allowed(client_b)
        self.assertTrue(allowed_a)
        self.assertTrue(allowed_b)

        # Client A is now throttled, but Client B second attempt also throttled
        self.assertFalse(limiter.is_allowed(client_a)[0])
        self.assertFalse(limiter.is_allowed(client_b)[0])


if __name__ == "__main__":
    unittest.main()
