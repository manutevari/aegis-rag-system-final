def test_local_provider_defaults_are_explicit(monkeypatch):
    from app.core.settings import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "local_auto")
    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "hash")
    monkeypatch.setenv("VECTOR_BACKEND", "chroma")
    monkeypatch.setenv("RERANK_PROVIDER", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("COHERE_API_KEY", "")
    monkeypatch.setenv("PINECONE_API_KEY", "")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.active_llm_provider == "local_auto"
    assert settings.local_llm_order == "ollama,llama_cpp,mistral_local"
    assert settings.local_orchestration_model == "llama3.1"
    assert settings.local_generation_model == "mistral"
    assert settings.ollama_model == "mistral"
    assert settings.llama_cpp_model == "local-model"
    assert settings.mistral_local_model == "mistral"
    assert settings.rag_embeddings_provider == "hash"
    assert settings.vector_backend == "chroma"
    assert settings.rerank_provider == "local"


def test_missing_hosted_keys_keep_safe_embedding_fallbacks(monkeypatch):
    from app.core.settings import get_settings
    from app.core.vector_store import LocalHashEmbeddings, get_embeddings

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "openai")
    get_settings.cache_clear()

    assert isinstance(get_embeddings(), LocalHashEmbeddings)


def test_missing_google_key_keeps_embedding_fallback(monkeypatch):
    from app.core.settings import get_settings
    from app.core.vector_store import LocalHashEmbeddings, get_embeddings

    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "google")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    get_settings.cache_clear()

    assert isinstance(get_embeddings(), LocalHashEmbeddings)
