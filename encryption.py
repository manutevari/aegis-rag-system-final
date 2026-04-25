"""Fernet symmetric encryption with env-var / file / auto-generate key hierarchy."""
import base64, logging, os, pathlib
from cryptography.fernet import Fernet, InvalidToken

logger   = logging.getLogger(__name__)
_KEY_PATH = pathlib.Path(os.getenv("ENCRYPTION_KEY_PATH", "/tmp/dg_rag.key"))
_fernet  = None

def _load_key() -> bytes:
    env = os.getenv("ENCRYPTION_KEY","")
    if env:
        try: return base64.urlsafe_b64decode(env.encode())
        except: pass
    if _KEY_PATH.exists(): return _KEY_PATH.read_bytes()
    k = Fernet.generate_key()
    _KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _KEY_PATH.write_bytes(k)
    logger.info("Generated new encryption key → %s", _KEY_PATH)
    return k

def _get() -> Fernet:
    global _fernet
    if _fernet is None: _fernet = Fernet(_load_key())
    return _fernet

def encrypt(data: str) -> bytes:
    return _get().encrypt(data.encode("utf-8"))

def decrypt(token: bytes) -> str:
    try: return _get().decrypt(token).decode("utf-8")
    except InvalidToken: raise ValueError("Decryption failed — key mismatch or expired token")
