import time
from typing import Any, Dict, Tuple

# Tiny TTL cache: { key: (expire_ts, value) }
_cache: Dict[str, Tuple[float, Any]] = {}


def set_cache(key: str, value: Any, ttl_seconds: int = 600) -> None:
    _cache[key] = (time.time() + ttl_seconds, value)


def get_cache(key: str):
    item = _cache.get(key)
    if not item:
        return None
    expire_ts, value = item
    if time.time() > expire_ts:
        _cache.pop(key, None)
        return None
    return value
