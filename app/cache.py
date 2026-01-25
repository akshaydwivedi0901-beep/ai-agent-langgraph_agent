import os
import json
import redis
from typing import Any

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)


class RedisCache:
    def __init__(self, prefix: str, ttl_seconds: int = 300):
        self.prefix = prefix
        self.ttl = ttl_seconds

    def _key(self, key: str) -> str:
        return f"{self.prefix}:{key}"

    def get(self, key: str) -> Any | None:
        value = redis_client.get(self._key(key))
        if value is None:
            return None
        return json.loads(value)

    def set(self, key: str, value: Any):
        redis_client.setex(
            self._key(key),
            self.ttl,
            json.dumps(value)
        )
