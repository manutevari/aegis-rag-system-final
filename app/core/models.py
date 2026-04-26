"""
Unified Model Manager — FINAL (OpenRouter → OpenAI fallback, strict models)
"""

import os
import time
import logging
from typing import Dict, Any
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


# ==============================
# 🔹 STRICT MODEL CONFIG
# ==============================

OPENROUTER_MODELS = [
    "nvidia/nemotron-3-super-120b",
    "meta-llama/llama-3.3-70b-instruct",
    "deepseek/deepseek-r1",
    "openai/gpt-oss-120b",
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

    if callable(key):
        key = key()

    if key is None:
        return None

    key = str(key).strip()
    if not key:
        return None

    if name == "OPENAI_API_KEY" and not key.startswith("sk-"):
        logger.warning("Invalid OPENAI_API_KEY format")
        return None

    if name == "OPENROUTER_API_KEY" and not (
        key.startswith("sk-or-") or key.startswith("sk-or-v1-")
    ):
        logger.warning("Invalid OPENROUTER_API_KEY format")
        return None

    return key


# ==============================
# 🔹 CLIENTS
# ==============================

def _get_openrouter_client():
    key = _safe_key("OPENROUTER_API_KEY", "OPENROUTER_KEY")
    if not key:
        return None

    return OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
    )


def _get_openai_client():
    key = _safe_key("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OPENAI_API_KEY")

    return OpenAI(api_key=key)


# ==============================
# 🔹 CORE INVOCATION (FINAL)
# ==============================

def invoke_llm(messages, model_override=None, max_tokens=512, temperature=0.1):
    """
    Routing:
    1. OpenRouter (strict models)
    2. Fallback → OpenAI
    """

    # ------------------------------
    # STEP 1: OPENROUTER
    # ------------------------------
    or_client = _get_openrouter_client()

    if or_client:
        for model in OPENROUTER_MODELS:
            try:
                logger.info(f"[OpenRouter] → {model}")

                res = or_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                return res

            except Exception as e:
                err = str(e)

                # 🔥 Backoff for rate/quota
                if "429" in err or "quota" in err.lower():
                    logger.warning(f"[OpenRouter] rate/quota hit: {model}")
                    time.sleep(1)
                    continue

                logger.warning(f"[OpenRouter] failed: {model} | {err}")
                continue

    else:
        logger.warning("⚠️ No OpenRouter key → skipping")

    # ------------------------------
    # STEP 2: OPENAI FALLBACK
    # ------------------------------
    logger.warning("⚠️ Switching → OpenAI fallback")

    oa_client = _get_openai_client()

    for model in OPENAI_MODELS:
        try:
            logger.info(f"[OpenAI] → {model}")

            res = oa_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            return res

        except Exception as e:
            logger.warning(f"[OpenAI] failed: {model} | {e}")
            continue

    # ------------------------------
    # FINAL FAIL
    # ------------------------------
    raise RuntimeError("❌ All providers exhausted (OpenRouter + OpenAI)")


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
