"""Runtime provider configuration for UI-supplied model keys."""

import os
from typing import Dict, Optional

from app.core.settings import get_settings

_PROVIDER_ALIASES = {
    "gemini": "gemini",
    "google": "gemini",
    "google-gemini": "gemini",
    "google gemini": "gemini",
    "openai": "openai",
    "gpt": "openai",
    "extractive": "extractive",
    "offline": "extractive",
    "local": "extractive",
}

_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "extractive": "",
}


def normalize_provider(provider: Optional[str]) -> str:
    """Normalize UI labels and env values to a supported provider id."""
    raw = (provider or "gemini").strip().lower()
    return _PROVIDER_ALIASES.get(raw, raw if raw in _DEFAULT_MODELS else "gemini")


def default_model_for_provider(provider: Optional[str]) -> str:
    """Return the default model name for a provider id or UI label."""
    return _DEFAULT_MODELS[normalize_provider(provider)]


def apply_runtime_model_config(
    provider: Optional[str],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, object]:
    """Apply UI-selected model config to process env for this Streamlit session."""
    normalized = normalize_provider(provider)
    selected_model = (model or default_model_for_provider(normalized)).strip()
    supplied_key = (api_key or "").strip()

    if normalized == "gemini":
        os.environ["LLM_PROVIDER"] = "gemini"
        if supplied_key:
            os.environ["GEMINI_API_KEY"] = supplied_key
        if selected_model:
            os.environ["GOOGLE_MODEL"] = selected_model

    elif normalized == "openai":
        os.environ["LLM_PROVIDER"] = "openai"
        if supplied_key:
            os.environ["OPENAI_API_KEY"] = supplied_key
        if selected_model:
            os.environ["OPENAI_MODEL"] = selected_model

    else:
        os.environ["LLM_PROVIDER"] = "extractive"
        selected_model = ""

    get_settings.cache_clear()

    return {
        "provider": normalized,
        "model": selected_model,
        "has_session_key": bool(supplied_key),
    }
