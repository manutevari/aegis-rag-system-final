def test_hosted_provider_defaults_are_explicit(monkeypatch):
    from app.core.settings import get_settings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.openai_model == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-large"
    assert settings.openai_embedding_dimensions == 3072
    assert settings.rerank_provider == "cohere"
    assert settings.cohere_rerank_model == "rerank-v3.5"
    assert settings.vector_backend == "pinecone"


def test_missing_hosted_keys_keep_safe_fallbacks(monkeypatch):
    from app.core.settings import get_settings
    from app.core.vector_store import LocalHashEmbeddings, get_embeddings

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("RAG_EMBEDDINGS_PROVIDER", "openai")
    get_settings.cache_clear()

    assert isinstance(get_embeddings(), LocalHashEmbeddings)
