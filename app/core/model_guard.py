import streamlit as st

ALLOWED_LLM_MODELS = {
    "llama3",
    "llama3.1",
    "mistral",
    "phi3",
}

ALLOWED_EMBED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "hash",
}


def get_llm_model():
    model = st.secrets.get("OLLAMA_MODEL", "llama3")

    if model not in ALLOWED_LLM_MODELS:
        raise ValueError(f"Unauthorized local model: {model}")

    return model


def get_embedding_model():
    model = st.secrets.get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    if model not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"Unauthorized embedding model: {model}")

    return model
