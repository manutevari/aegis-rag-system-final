"""
Unified Model Manager — Sync-safe, production-ready

Eliminates:
- async/sync mismatch
- model object vs string confusion
- scattered LLM creation
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
# 🔹 Config Loader
# ==============================

def _get_config() -> Dict[str, Any]:
    model = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)

    if model not in ALLOWED_CHAT_MODELS:
        raise ValueError(f"❌ Unauthorized chat model: {model}")

    api_key = os.getenv("OPENAI_API_KEY")

    if not isinstance(api_key, str) or not api_key:
        raise ValueError("❌ OPENAI_API_KEY must be a valid string")

    return {
        "model": model,
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.1")),
        "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "1024")),
        "api_key": api_key,
    }


# ==============================
# 🔹 Primary LLM
# ==============================

def get_primary_llm() -> ChatOpenAI:
    cfg = _get_config()

    return ChatOpenAI(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        api_key=cfg["api_key"],
    )


# ==============================
# 🔹 Fallback LLM (OpenRouter)
# ==============================

def get_fallback_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("FALLBACK_MODEL", "meta-llama/llama-3-70b-instruct"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


# ==============================
# 🔹 Unified Invocation
# ==============================

def invoke_llm(messages):
    """
    Single entry point for ALL LLM calls
    """

    primary = get_primary_llm()

    try:
        return primary.invoke(messages)

    except Exception as e:
        logger.warning(f"Primary failed: {e}")

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
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
