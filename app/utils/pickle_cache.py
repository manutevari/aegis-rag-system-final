"""Pickle cache: TTL + LRU eviction. Keys are SHA-256 hashes. Values are encrypted bytes."""
import hashlib, logging, os, pathlib, pickle, time
from typing import Optional

logger    = logging.getLogger(__name__)
CACHE_DIR = pathlib.Path(os.getenv("CACHE_DIR", "/tmp/dg_rag_cache"))
TTL       = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX       = int(os.getenv("CACHE_MAX_ENTRIES", "1000"))

class PickleCache:
    def __init__(self):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> pathlib.Path:
        h = hashlib.sha256(key.encode()).hexdigest()
        return CACHE_DIR / h[:2] / f"{h}.pkl"

    def get(self, key: str) -> Optional[bytes]:
        p = self._path(key)
        if not p.exists(): return None
        try:
            e = pickle.loads(p.read_bytes())
            if time.time() - e["ts"] > TTL:
                p.unlink(missing_ok=True); return None
            return e["v"]
        except: return None

    def set(self, key: str, value: bytes) -> None:
        self._evict()
        p = self._path(key); p.parent.mkdir(parents=True, exist_ok=True)
        try: p.write_bytes(pickle.dumps({"ts": time.time(), "v": value}))
        except Exception as e: logger.warning("Cache write error: %s", e)

    def clear(self) -> int:
        n = 0
        for f in CACHE_DIR.rglob("*.pkl"): f.unlink(missing_ok=True); n += 1
        return n

    def stats(self) -> dict:
        fs = list(CACHE_DIR.rglob("*.pkl"))
        b  = sum(f.stat().st_size for f in fs)
        return {"entries": len(fs), "size_mb": round(b/1_048_576, 2)}

    def _evict(self):
        fs = sorted(CACHE_DIR.rglob("*.pkl"), key=lambda f: f.stat().st_mtime)
        while len(fs) >= MAX:
            try: fs.pop(0).unlink(missing_ok=True)
            except: break
