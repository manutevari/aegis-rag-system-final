"""
Model Registry — Safe, validated model access layer
Fix: Ensures chat model returns STRING (not object) to avoid .lower() crash
"""

import os
from typing import Optional
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# 🔹 Import middleware (safe optional)
try:
    from middleware import orchestrator_middleware
except Exception:
    orchestrator_middleware = None


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
# 🔹 Chat Model (PRIMARY FIX)
# ==============================

def get_chat_model() -> str:
    """
    Returns ONLY model name (string).
    This prevents `.lower()` crashes inside LangChain/OpenAI wrappers.
    """
    model_name = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)

    if model_name not in ALLOWED_CHAT_MODELS:
        raise ValueError(f"❌ Unauthorized chat model: {model_name}")

    return model_name


# ==============================
# 🔹 Optional: Full LLM Object (SAFE)
# ==============================

def get_chat_llm(temperature: float = 0.1, max_tokens: int = 1024):
    """
    Returns actual Chat model object (ONLY use if needed explicitly).
    Safe wrapper with middleware support.
    """
    model_name = get_chat_model()

    llm = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 🔹 Attach middleware if supported
    if orchestrator_middleware:
        try:
            llm.middleware = [orchestrator_middleware]
        except Exception:
            pass

    return llm


# ==============================
# 🔹 Embedding Model
# ==============================

def get_embed_model():
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if model_name not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"❌ Unauthorized embedding model: {model_name}")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )