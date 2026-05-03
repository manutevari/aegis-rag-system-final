def _clear_settings_cache():
    from app.core.settings import get_settings

    get_settings.cache_clear()


def test_decision_manager_selects_ollama_first_when_working(monkeypatch):
    import app.core.llm_decision_manager as manager

    monkeypatch.setenv("LLM_PROVIDER", "local_auto")
    monkeypatch.setenv("LOCAL_GENERATION_MODEL", "mistral")
    _clear_settings_cache()

    def fake_json_request(url, payload=None, timeout=2.0):
        assert payload is None
        if url.endswith("/api/tags"):
            return {"models": [{"name": "mistral:latest"}]}
        raise RuntimeError(f"unexpected url {url}")

    monkeypatch.setattr(manager, "_json_request", fake_json_request)

    llm, decision = manager.select_local_llm(node="generator")

    assert llm.provider == "ollama"
    assert decision.provider == "ollama"
    assert decision.model == "mistral"
    assert decision.skipped == []


def test_decision_manager_skips_ollama_to_llama_cpp(monkeypatch):
    import app.core.llm_decision_manager as manager

    monkeypatch.setenv("LLM_PROVIDER", "local_auto")
    monkeypatch.setenv("LOCAL_LLM_ORDER", "ollama,llama_cpp,mistral_local")
    _clear_settings_cache()

    def fake_json_request(url, payload=None, timeout=2.0):
        assert payload is None
        if url.endswith("/api/tags"):
            raise RuntimeError("ollama down")
        if url.endswith("/health") and ":8080" in url:
            return {}
        raise RuntimeError(f"unexpected url {url}")

    monkeypatch.setattr(manager, "_json_request", fake_json_request)

    llm, decision = manager.select_local_llm(node="generator")

    assert llm.provider == "llama_cpp"
    assert decision.provider == "llama_cpp"
    assert decision.skipped[0]["provider"] == "ollama"


def test_decision_manager_returns_extract_fallback_when_all_unavailable(monkeypatch):
    import app.core.llm_decision_manager as manager

    monkeypatch.setenv("LLM_PROVIDER", "local_auto")
    _clear_settings_cache()
    monkeypatch.setattr(manager, "_json_request", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("down")))

    llm, decision = manager.select_local_llm(node="generator")

    assert llm is None
    assert decision.provider == "extractive"
    assert [item["provider"] for item in decision.skipped] == ["ollama", "llama_cpp", "mistral_local"]


def test_decision_manager_uses_orchestration_model_for_router(monkeypatch):
    import app.core.llm_decision_manager as manager

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LOCAL_ORCHESTRATION_MODEL", "llama3.1")
    _clear_settings_cache()
    monkeypatch.setattr(manager, "_json_request", lambda *args, **kwargs: {"models": [{"name": "llama3.1:latest"}]})

    llm, decision = manager.select_local_llm(node="router")

    assert llm.provider == "ollama"
    assert decision.role == "orchestration"
    assert decision.model == "llama3.1"
