def _clear_settings_cache():
    from app.core.settings import get_settings

    get_settings.cache_clear()


def test_default_openai_llm_falls_back_without_api_key(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("OPENAI_API_KEY", "")
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


def test_gemini_provider_falls_back_without_api_key(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_gemini_provider_uses_gemini_api_key(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.GooglePolicyModel)
    assert llm.model == "gemini-2.5-flash"


def test_ollama_provider_falls_back_when_server_unreachable(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(models, "_ollama_server_available", lambda base_url: False)
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)
