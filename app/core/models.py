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
# 🔹 SAFE ENV LOADER
# ==============================

def _safe_key(name: str) -> str:
    key = os.getenv(name)

    # 🔥 handle weird deployment cases
    if callable(key):
        key = key()

    if not isinstance(key, str):
        key = str(key)

    if not key or "sk-" not in key:
        raise ValueError(f"❌ Invalid {name}")

    return key


# ==============================
# 🔹 Config Loader
# ==============================

def _get_config(model_override=None) -> Dict[str, Any]:
    model = model_override or os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)

    if model not in ALLOWED_CHAT_MODELS:
        logger.warning(f"Using non-whitelisted model: {model}")

    return {
        "model": model,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "api_key": _safe_key("OPENAI_API_KEY"),
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

def get_fallback_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("FALLBACK_MODEL", "mistralai/mixtral-8x7b"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        base_url="https://openrouter.ai/api/v1",
        api_key=_safe_key("OPENROUTER_API_KEY"),
        streaming=False,  # 🔥 critical
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
        logger.warning(f"Primary failed → switching to fallback: {e}")

        fallback = get_fallback_llm()

        try:
            return fallback.invoke(messages)
        except Exception as e2:
            logger.error(f"Fallback failed: {e2}")
            raise RuntimeError("Both LLMs failed")


# ==============================
# 🔹 Embeddings
# ==============================

def get_embed_model():
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if model_name not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"❌ Unauthorized embedding model: {model_name}")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=_safe_key("OPENAI_API_KEY"),
    )
