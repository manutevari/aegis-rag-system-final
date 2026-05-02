def test_default_llm_uses_extractive_model_without_ollama(monkeypatch):
    import app.core.models as models

    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_PROVIDER", raising=False)

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_ollama_provider_falls_back_when_server_unreachable(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(models, "_ollama_server_available", lambda base_url: False)

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)
