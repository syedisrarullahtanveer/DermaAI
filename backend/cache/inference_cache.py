"""
In-memory LRU inference cache with TTL eviction for duplicate image hash queries.
"""
import time
from collections import OrderedDict

class InferenceCache:
    def __init__(self, max_size: int = 256, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        val, ts = self.cache[key]
        if time.time() - ts > self.ttl_seconds:
            del self.cache[key]
            return None
        self.cache.move_to_end(key)
        return val

    def set(self, key: str, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = (value, time.time())
