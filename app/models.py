import os

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


def get_chat_model():
    model = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)
    if model not in ALLOWED_CHAT_MODELS:
        raise ValueError(f"❌ Unauthorized chat model: {model}")
    return model


def get_embed_model():
    model = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
    if model not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"❌ Unauthorized embedding model: {model}")
    return model
