"""
Unified Model Manager — FINAL (stable + retry-aware)
"""

import os
import logging
from typing import Dict, Any
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

logger = logging.getLogger(__name__)


# ==============================
# 🔹 Allowed Models
# ==============================

ALLOWED_CHAT_MODELS = {
    "gpt-5-nano",
    "gpt-5-mini",
    "gpt-4o-mini",
}

ALLOWED_EMBED_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
}

DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


# ==============================
# 🔹 SAFE ENV LOADER (UPDATED)
# ==============================

def _safe_key(name: str, alt_name: str = None) -> str | None:
    """
    Returns a clean string key.
    Accepts primary name and optional alternative name.
    Returns None if missing/invalid (so we can skip fallback gracefully).
    """
    key = os.getenv(name) or (os.getenv(alt_name) if alt_name else None)

    if callable(key):
        key = key()

    if key is None:
        return None

    if not isinstance(key, str):
        key = str(key)

    key = key.strip()
    if not key:
        return None

    # Provider-specific validation
    if name == "OPENAI_API_KEY" and not key.startswith("sk-"):
        logger.warning("OPENAI_API_KEY format looks invalid")
        return None

    # Accept both formats for OpenRouter
    if name == "OPENROUTER_API_KEY" and not (
        key.startswith("sk-or-") or key.startswith("sk-or-v1-")
    ):
        logger.warning("OPENROUTER key format looks invalid")
        return None

    return key


# ==============================
# 🔹 Config Loader
# ==============================

def _get_config(model_override=None) -> Dict[str, Any]:
    model = model_override or os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)

    if model not in ALLOWED_CHAT_MODELS:
        logger.warning(f"Using non-whitelisted model: {model}")

    api_key = _safe_key("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("Missing/invalid OPENAI_API_KEY")

    return {
        "model": model,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "api_key": api_key,
    }


# ==============================
# 🔹 Primary LLM
# ==============================

def get_primary_llm(model_override=None) -> ChatOpenAI:
    cfg = _get_config(model_override)

    return ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        api_key=cfg["api_key"],
        streaming=False,  # 🔥 critical fix
    )


# ==============================
# 🔹 Fallback LLM (OpenRouter)
# ==============================

def get_fallback_llm() -> ChatOpenAI | None:
    # ✅ Accept BOTH env names
    key = _safe_key("OPENROUTER_API_KEY", alt_name="OPENROUTER_KEY")

    if not key:
        logger.warning("No valid OpenRouter key found → skipping fallback")
        return None

    return ChatOpenAI(
        model=os.getenv("FALLBACK_MODEL", "mistralai/mixtral-8x7b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        streaming=False,
    )


# ==============================
# 🔹 Unified Invocation
# ==============================

def invoke_llm(messages, model_override=None):
    """
    Single entry point for ALL LLM calls
    Supports retry controller model override
    """

    primary = get_primary_llm(model_override)

    try:
        return primary.invoke(messages)

    except Exception as e:
        logger.warning(f"Primary failed → {e}")

        fallback = get_fallback_llm()

        if not fallback:
            raise RuntimeError("Primary failed and no fallback available")

        return fallback.invoke(messages)


# ==============================
# 🔹 Embeddings
# ==============================

def get_embed_model():
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if model_name not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"❌ Unauthorized embedding model: {model_name}")

    key = _safe_key("OPENAI_API_KEY")

    if not key:
        raise ValueError("Missing/invalid OPENAI_API_KEY")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=key,
    )
