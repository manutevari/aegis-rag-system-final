import os
import time
import logging
from typing import Optional
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ==============================
# 🔹 STRICT MODEL CONFIG (CLEANED)
# ==============================

# ❌ Removed broken models like nemotron
OPENROUTER_MODELS = [
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1",
    "qwen/qwen3-next-80b",
]

OPENAI_MODELS = [
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-4o-mini",
]

OPENAI_EMBED_MODELS = [
    "text-embedding-3-large",
    "text-embedding-3-small",
]

DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# ==============================
# 🔹 SAFE KEY LOADER
# ==============================

def _safe_key(name: str, alt_name: str = None) -> str | None:
    key = os.getenv(name) or (os.getenv(alt_name) if alt_name else None)

    if key is None:
        return None

    key = str(key).strip()
    if not key:
        return None

    return key

# ==============================
# 🔹 CLIENTS
# ==============================

def _or_client():
    key = _safe_key("OPENROUTER_API_KEY", "OPENROUTER_KEY")
    if not key:
        return None
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

def _oa_client():
    key = _safe_key("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)

# ==============================
# 🔹 CACHE (SMART FAST PATH)
# ==============================

_LAST_GOOD_OR: Optional[str] = None
_LAST_GOOD_OA: Optional[str] = None

# ==============================
# 🔹 ERROR DETECTORS
# ==============================

def _is_quota(err: str):
    err = err.lower()
    return "429" in err or "quota" in err or "rate" in err

def _is_invalid(err: str):
    err = err.lower()
    return "400" in err or "404" in err or "not found" in err

# ==============================
# 🔹 CORE INVOCATION (SMART)
# ==============================

def _call(client, model, messages, max_tokens, temperature):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

def invoke_llm(messages, model_override=None, max_tokens=120, temperature=0):
    global _LAST_GOOD_OR, _LAST_GOOD_OA

    # -----------------------------
    # 0. OVERRIDE
    # -----------------------------
    if model_override:
        try:
            client = _oa_client() or _or_client()
            return _call(client, model_override, messages, max_tokens, temperature)
        except Exception as e:
            logger.warning(f"[override failed] {e}")

    # -----------------------------
    # 1. FAST PATH (CACHE)
    # -----------------------------
    if _LAST_GOOD_OR:
        try:
            logger.info(f"[OR fast] {_LAST_GOOD_OR}")
            return _call(_or_client(), _LAST_GOOD_OR, messages, max_tokens, temperature)
        except Exception:
            pass

    if _LAST_GOOD_OA:
        try:
            logger.info(f"[OA fast] {_LAST_GOOD_OA}")
            return _call(_oa_client(), _LAST_GOOD_OA, messages, max_tokens, temperature)
        except Exception:
            pass

    # -----------------------------
    # 2. OPENROUTER PROBE
    # -----------------------------
    or_client = _or_client()
    if or_client:
        for m in OPENROUTER_MODELS:
            try:
                logger.info(f"[OR try] {m}")
                res = _call(or_client, m, messages, max_tokens, temperature)
                _LAST_GOOD_OR = m
                return res

            except Exception as e:
                err = str(e)

                if _is_invalid(err):
                    logger.warning(f"[OR invalid] {m}")
                    continue

                if _is_quota(err):
                    logger.warning(f"[OR quota] {m}")
                    time.sleep(0.5)
                    continue

                logger.warning(f"[OR fail] {m} | {err}")
                continue

    # -----------------------------
    # 3. OPENAI FALLBACK
    # -----------------------------
    oa_client = _oa_client()
    if oa_client:
        for m in OPENAI_MODELS:
            try:
                logger.info(f"[OA try] {m}")
                res = _call(oa_client, m, messages, max_tokens, temperature)
                _LAST_GOOD_OA = m
                return res

            except Exception as e:
                err = str(e)

                if _is_quota(err):
                    logger.warning(f"[OA quota] {m}")
                    time.sleep(0.5)
                    continue

                logger.warning(f"[OA fail] {m} | {err}")
                continue

    # -----------------------------
    # 4. HARD STOP (NO LOOP)
    # -----------------------------
    logger.error("❌ All models exhausted")

    return type(
        "LLMResponse",
        (),
        {"content": "⚠️ All models unavailable. Check API keys or quota."},
    )()


# ==============================
# 🔹 EMBEDDINGS (STRICT)
# ==============================

def get_embed_model():
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if model_name not in OPENAI_EMBED_MODELS:
        raise ValueError(f"Unauthorized embedding model: {model_name}")

    key = _safe_key("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=key,
    )
