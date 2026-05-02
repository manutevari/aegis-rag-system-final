def _clear_settings_cache():
    from app.core.settings import get_settings

    get_settings.cache_clear()


def test_default_openai_llm_falls_back_without_api_key(monkeypatch):
    import app.core.models as models

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_explicit_extractive_provider_uses_local_model(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_ollama_provider_falls_back_when_server_unreachable(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(models, "_ollama_server_available", lambda base_url: False)
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)
