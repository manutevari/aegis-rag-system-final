import streamlit as st

ALLOWED_LLM_MODELS = {
    "llama3",
    "llama3.1",
    "mistral",
    "mixtral",
    "phi3",
    "local-model",
    "extractive",
}

ALLOWED_EMBED_MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "hash",
}

ALLOWED_RERANK_MODELS = {
    "lexical",
    "BAAI/bge-reranker-base",
}


def get_llm_model():
    model = st.secrets.get("LOCAL_GENERATION_MODEL", st.secrets.get("OLLAMA_MODEL", "mistral"))

    if model not in ALLOWED_LLM_MODELS:
        raise ValueError(f"Unauthorized local model: {model}")

    return model


def get_embedding_model():
    model = st.secrets.get("LOCAL_EMBED_MODEL", "hash")

    if model not in ALLOWED_EMBED_MODELS:
        raise ValueError(f"Unauthorized embedding model: {model}")

    return model


def get_rerank_model():
    model = st.secrets.get("RERANK_MODEL", "lexical")

    if model not in ALLOWED_RERANK_MODELS:
        raise ValueError(f"Unauthorized rerank model: {model}")

    return model
