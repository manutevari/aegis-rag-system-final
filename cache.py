# =============================================================================
# AEGIS — CACHING LAYER
# cache.py
#
# Lecture point: "Introduce caching layers (e.g., Redis) to reduce latency
# and operational costs." (Operational & Environment Reminders #3)
#
# Design:
#   • Primary: Redis (if REDIS_URL env var is set and redis-py is installed)
#   • Fallback: in-process LRU dict (no extra infra needed for dev/testing)
#   • Cache key: SHA-256 of (query + category + alpha) — deterministic,
#     collision-resistant, and independent of object identity
#   • TTL: 3600 s (1 hour) for Redis entries; unlimited for in-process LRU
#   • Cached value: the full run_query() result dict (JSON-serialisable)
#
# Pydantic enforcement:
#   CacheEntry validates the stored payload so stale / corrupt cache entries
#   are rejected on read rather than propagating bad data downstream.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

__all__ = ["get_cache", "AegisCache"]

CACHE_TTL = int(os.getenv("AEGIS_CACHE_TTL", "3600"))   # seconds


# ---------------------------------------------------------------------------
# Pydantic cache entry model
# ---------------------------------------------------------------------------

class CacheEntry(BaseModel):
    """Validated wrapper for a cached pipeline result."""

    query:      str        = Field(..., description="Original user query")
    result:     dict       = Field(..., description="Full run_query() result dict")
    created_at: float      = Field(default_factory=time.time)
    ttl:        int        = Field(default=CACHE_TTL)

    @field_validator("result")
    @classmethod
    def result_has_answer(cls, v: dict) -> dict:
        if "answer" not in v:
            raise ValueError("cached result must contain 'answer' key")
        return v

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl


# ---------------------------------------------------------------------------
# In-process LRU fallback (thread-safe enough for Streamlit single-process)
# ---------------------------------------------------------------------------

class _InProcessCache:
    """Simple dict-based LRU cache with TTL. Used when Redis is unavailable."""

    MAX_SIZE = 256

    def __init__(self) -> None:
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> dict | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if entry.is_expired:
            del self._store[key]
            return None
        return entry.result

    def set(self, key: str, query: str, result: dict) -> None:
        try:
            entry = CacheEntry(query=query, result=result)
            if len(self._store) >= self.MAX_SIZE:
                # Evict oldest
                oldest = min(self._store, key=lambda k: self._store[k].created_at)
                del self._store[oldest]
            self._store[key] = entry
        except Exception:
            pass   # Never let cache writes crash the pipeline

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Redis cache (optional)
# ---------------------------------------------------------------------------

class _RedisCache:
    """Redis-backed cache. Falls back to _InProcessCache on any error."""

    def __init__(self, redis_url: str) -> None:
        import redis   # type: ignore
        self._client = redis.from_url(redis_url, decode_responses=True)
        self._prefix = "aegis:"

    def get(self, key: str) -> dict | None:
        try:
            raw = self._client.get(self._prefix + key)
            if raw is None:
                return None
            data = json.loads(raw)
            entry = CacheEntry(**data)
            if entry.is_expired:
                self._client.delete(self._prefix + key)
                return None
            return entry.result
        except Exception:
            return None

    def set(self, key: str, query: str, result: dict) -> None:
        try:
            entry   = CacheEntry(query=query, result=result)
            payload = entry.model_dump_json()
            self._client.setex(self._prefix + key, CACHE_TTL, payload)
        except Exception:
            pass

    def clear(self) -> None:
        try:
            keys = self._client.keys(self._prefix + "*")
            if keys:
                self._client.delete(*keys)
        except Exception:
            pass

    @property
    def size(self) -> int:
        try:
            return len(self._client.keys(self._prefix + "*"))
        except Exception:
            return -1


# ---------------------------------------------------------------------------
# Public AegisCache façade
# ---------------------------------------------------------------------------

class AegisCache:
    """
    Unified cache façade. Automatically selects Redis or in-process backend.

    Usage:
        cache = AegisCache()
        key   = cache.make_key(query, category, alpha)
        hit   = cache.get(key)
        if hit is None:
            result = run_query(query)
            cache.set(key, query, result)
    """

    def __init__(self) -> None:
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            try:
                self._backend = _RedisCache(redis_url)
                self._backend_name = "redis"
            except Exception:
                self._backend = _InProcessCache()
                self._backend_name = "in-process"
        else:
            self._backend = _InProcessCache()
            self._backend_name = "in-process"

    @staticmethod
    def make_key(query: str, category: str | None = None,
                 alpha: float = 0.15) -> str:
        """
        Deterministic SHA-256 cache key.
        Includes category and alpha so different pipeline configs get
        different cache slots.
        """
        raw = json.dumps({"q": query.strip().lower(),
                          "cat": category, "alpha": round(alpha, 3)},
                         sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> dict | None:
        return self._backend.get(key)

    def set(self, key: str, query: str, result: dict) -> None:
        self._backend.set(key, query, result)

    def clear(self) -> None:
        self._backend.clear()

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def size(self) -> int:
        return self._backend.size


# Module-level singleton — import and reuse across requests
_cache_singleton: AegisCache | None = None

def get_cache() -> AegisCache:
    """Return the module-level AegisCache singleton."""
    global _cache_singleton
    if _cache_singleton is None:
        _cache_singleton = AegisCache()
    return _cache_singleton
