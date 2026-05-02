import streamlit as st

ALLOWED_LLM_MODELS = {
    "gpt-4o-mini",
    "llama3",
    "llama3.1",
    "mistral",
    "phi3",
}

ALLOWED_EMBED_MODELS = {
    "text-embedding-3-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "hash",
}

ALLOWED_RERANK_MODELS = {
    "rerank-v3.5",
    "BAAI/bge-reranker-base",
}


def get_llm_model():
    model = st.secrets.get("OPENAI_MODEL", st.secrets.get("OLLAMA_MODEL", "gpt-4o-mini"))

    if model not in ALLOWED_LLM_MODELS:
        raise ValueError(f"Unauthorized model: {model}")

    return model


def get_embedding_model():
    model = st.secrets.get("OPENAI_EMBEDDING_MODEL", st.secrets.get("LOCAL_EMBED_MODEL", "text-embedding-3-large"))

    if model not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"Unauthorized embedding model: {model}")

    return model


def get_rerank_model():
    model = st.secrets.get("COHERE_RERANK_MODEL", st.secrets.get("RERANK_MODEL", "rerank-v3.5"))

    if model not in ALLOWED_RERANK_MODELS:
        raise ValueError(f"Unauthorized rerank model: {model}")

    return model
