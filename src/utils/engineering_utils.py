"""Engineering utility functions for DermaAI.

This module contains small, reusable helpers that improve
performance, reliability, and developer ergonomics.
"""

from functools import wraps
from typing import Any, Callable, Dict, Tuple
import time

# In‑memory cache shared by all memoized functions in this process.
_memoize_cache: Dict[Tuple[Callable, Tuple[Any, ...], frozenset], Any] = {}

def memoize(func: Callable) -> Callable:
    """Simple memoization decorator.

    Caches results based on positional arguments and a frozen set of
    keyword‑argument items. Only works for hashable arguments.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        key = (func, args, frozenset(kwargs.items()))
        if key in _memoize_cache:
            return _memoize_cache[key]
        result = func(*args, **kwargs)
        _memoize_cache[key] = result
        return result
    return wrapper

def timing(func: Callable) -> Callable:
    """Timing decorator that prints execution time.

    Returns the original function's result.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMING] {func.__name__} took {elapsed:.6f}s")
        return result
    return wrapper

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert *value* to float safely.

    Returns *default* if conversion fails or value is ``None``/empty.
    """
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default
