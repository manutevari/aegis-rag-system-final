import os
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings

# 🔹 Import middleware (adjust path if needed)
from middleware import orchestrator_middleware


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
# 🔹 Chat Model (FIXED)
# ==============================

def get_chat_model():
    model_name = os.getenv("LLM_MODEL", DEFAULT_CHAT_MODEL)

    if model_name not in ALLOWED_CHAT_MODELS:
        raise ValueError(f"❌ Unauthorized chat model: {model_name}")

    # Initialize actual model (NOT string)
    model = init_chat_model(
        model=model_name,
        model_provider="openai",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # 🔹 Attach middleware safely
    try:
        model.middleware = [orchestrator_middleware]
    except Exception:
        # Fallback for versions that don't support middleware attr
        pass

    return model


# ==============================
# 🔹 Embedding Model (FIXED)
# ==============================

def get_embed_model():
    model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)

    if model_name not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"❌ Unauthorized embedding model: {model_name}")

    return OpenAIEmbeddings(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
