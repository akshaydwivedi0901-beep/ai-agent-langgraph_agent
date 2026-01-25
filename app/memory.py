import os
import json
import redis
from typing import List

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

MAX_HISTORY = 6
MEMORY_TTL_SECONDS = 3600  # 1 hour per session


def _key(session_id: str) -> str:
    return f"chat:memory:{session_id}"


def get_history(session_id: str) -> List[dict]:
    raw = redis_client.lrange(_key(session_id), -MAX_HISTORY, -1)
    return [json.loads(x) for x in raw]


def append_message(session_id: str, role: str, content: str):
    msg = json.dumps({"role": role, "content": content})

    redis_client.rpush(_key(session_id), msg)
    redis_client.expire(_key(session_id), MEMORY_TTL_SECONDS)
