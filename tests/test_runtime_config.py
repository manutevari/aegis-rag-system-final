import os


def test_apply_runtime_gemini_config_sets_session_env(monkeypatch):
    from app.core.runtime_config import apply_runtime_model_config
    from app.core.settings import get_settings

    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    monkeypatch.setenv("LLM_PROVIDER", "extractive")
    get_settings.cache_clear()

    result = apply_runtime_model_config(
        "gemini",
        api_key="test-gemini-key",
        model="gemini-2.5-flash",
    )
    settings = get_settings()

    assert result == {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "has_session_key": True,
    }
    assert os.environ["LLM_PROVIDER"] == "gemini"
    assert os.environ["GEMINI_API_KEY"] == "test-gemini-key"
    assert os.environ["GOOGLE_MODEL"] == "gemini-2.5-flash"
    assert settings.active_llm_provider == "gemini"
    assert settings.google_key == "test-gemini-key"


def test_apply_runtime_extractive_config_does_not_require_key(monkeypatch):
    from app.core.runtime_config import apply_runtime_model_config
    from app.core.settings import get_settings

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    get_settings.cache_clear()

    result = apply_runtime_model_config("extractive", api_key="", model="ignored")
    settings = get_settings()

    assert result["provider"] == "extractive"
    assert result["model"] == ""
    assert result["has_session_key"] is False
    assert settings.active_llm_provider == "extractive"
