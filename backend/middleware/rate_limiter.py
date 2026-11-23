"""Sliding Window In-Memory Rate Limiter for Protecting Inference Endpoints."""

import time
from collections import defaultdict, deque
from threading import Lock
from typing import Dict, Tuple, Optional


class InMemoryRateLimiter:
    """Sliding window counter rate limiter per client IP address."""

    def __init__(self, max_requests: int = 60, window_seconds: float = 60.0) -> None:
        """Initialize rate limiter.

        Args:
            max_requests: Maximum number of allowed requests in the sliding time window.
            window_seconds: Time window duration in seconds.
        """
        if max_requests <= 0:
            raise ValueError("max_requests must be > 0")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")

        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: Dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, client_id: str) -> Tuple[bool, int, float]:
        """Check if request from client_id is permitted under the rate limit.

        Args:
            client_id: Unique client identifier (e.g. IP address).

        Returns:
            Tuple[bool, int, float]: (is_allowed, remaining_requests, reset_after_seconds).
        """
        current_time = time.time()
        expiry_time = current_time - self.window_seconds

        with self._lock:
            timestamps = self._clients[client_id]

            # Evict timestamps outside the sliding window
            while timestamps and timestamps[0] <= expiry_time:
                timestamps.popleft()

            if len(timestamps) < self.max_requests:
                timestamps.append(current_time)
                remaining = self.max_requests - len(timestamps)
                return True, remaining, 0.0

            # Rate limit exceeded: calculate time until oldest request leaves window
            oldest = timestamps[0]
            retry_after = max(0.0, (oldest + self.window_seconds) - current_time)
            return False, 0, retry_after

    def reset(self) -> None:
        """Clear all client tracking history."""
        with self._lock:
            self._clients.clear()
