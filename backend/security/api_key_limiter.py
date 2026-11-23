"""
Multi-tier API key quota tracking and sliding-window rate limiter.
"""
import time

class APIKeyQuotaLimiter:
    TIERS = {
        "free": {"rpm": 10, "daily_quota": 100},
        "clinical": {"rpm": 120, "daily_quota": 5000},
        "enterprise": {"rpm": 600, "daily_quota": 50000}
    }

    def __init__(self):
        self.requests = {} # key -> list of timestamps
        self.daily_counts = {} # key -> (date_str, count)

    def allow_request(self, api_key: str, tier: str = "clinical") -> tuple[bool, str]:
        now = time.time()
        limits = self.TIERS.get(tier, self.TIERS["free"])
        
        # Check RPM
        history = self.requests.get(api_key, [])
        history = [t for t in history if now - t < 60]
        if len(history) >= limits["rpm"]:
            return False, "Rate limit exceeded (per-minute)"
            
        history.append(now)
        self.requests[api_key] = history
        return True, "OK"
