def _clear_settings_cache():
    from app.core.settings import get_settings

    get_settings.cache_clear()


def _fallback_decision(node="generator"):
    from app.core.llm_decision_manager import LocalModelDecision

    return LocalModelDecision(node=node, role="generation", provider="extractive", skipped=[])


def test_default_local_auto_falls_back_when_runtimes_unavailable(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "local_auto")
    monkeypatch.setattr(models, "select_local_llm", lambda **kwargs: (None, _fallback_decision(kwargs.get("node", "generator"))))
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_explicit_extractive_provider_uses_local_model(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_hosted_provider_is_skipped_even_with_api_key(monkeypatch):
    import app.core.models as models

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    _clear_settings_cache()

    llm = models.get_llm()

    assert isinstance(llm, models.LocalPolicyModel)


def test_ollama_provider_uses_decision_manager_selection(monkeypatch):
    import app.core.models as models

    class FakeLocalLLM:
        provider = "ollama"
        model = "mistral"
        decision = {"provider": "ollama", "model": "mistral"}

        def invoke(self, messages):
            raise AssertionError("selection test should not invoke")

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setattr(models, "select_local_llm", lambda **kwargs: (FakeLocalLLM(), _fallback_decision(kwargs.get("node", "generator"))))
    _clear_settings_cache()

    llm = models.get_llm(node="generator")

    assert isinstance(llm, FakeLocalLLM)
    assert llm.provider == "ollama"
