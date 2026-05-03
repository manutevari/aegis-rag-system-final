"""Runtime provider configuration for UI-supplied model keys."""
"""Runtime configuration helpers for Streamlit local model controls."""

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
    "local auto": "local_auto",
    "local_auto": "local_auto",
    "auto": "local_auto",
    "ollama": "ollama",
    "llama.cpp": "llama_cpp",
    "llama_cpp": "llama_cpp",
    "mistral local": "mistral_local",
    "mistral_local": "mistral_local",
    "extractive": "extractive",
}


def normalize_local_provider(provider: Optional[str]) -> str:
    raw = (provider or "local_auto").strip().lower().replace("-", " ")
    return _PROVIDER_ALIASES.get(raw, "local_auto")


def apply_local_runtime_config(
    provider: Optional[str],
    ollama_base_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
    llama_cpp_base_url: Optional[str] = None,
    llama_cpp_model: Optional[str] = None,
    mistral_local_base_url: Optional[str] = None,
    mistral_local_model: Optional[str] = None,
    local_orchestration_model: Optional[str] = None,
    local_generation_model: Optional[str] = None,
) -> Dict[str, str]:
    """Apply local runtime choices to process env for this Streamlit session."""
    normalized = normalize_local_provider(provider)
    os.environ["LLM_PROVIDER"] = normalized

    _setenv_if_value("OLLAMA_BASE_URL", ollama_base_url)
    _setenv_if_value("OLLAMA_MODEL", ollama_model)
    _setenv_if_value("LLAMA_CPP_BASE_URL", llama_cpp_base_url)
    _setenv_if_value("LLAMA_CPP_MODEL", llama_cpp_model)
    _setenv_if_value("MISTRAL_LOCAL_BASE_URL", mistral_local_base_url)
    _setenv_if_value("MISTRAL_LOCAL_MODEL", mistral_local_model)
    _setenv_if_value("LOCAL_ORCHESTRATION_MODEL", local_orchestration_model)
    _setenv_if_value("LOCAL_GENERATION_MODEL", local_generation_model)

    get_settings.cache_clear()

    return {
        "provider": normalized,
        "model": selected_model,
        "has_session_key": bool(supplied_key),
    }
        "ollama_model": os.getenv("OLLAMA_MODEL", "mistral"),
        "llama_cpp_model": os.getenv("LLAMA_CPP_MODEL", "local-model"),
        "mistral_local_model": os.getenv("MISTRAL_LOCAL_MODEL", "mistral"),
        "local_orchestration_model": os.getenv("LOCAL_ORCHESTRATION_MODEL", "llama3.1"),
        "local_generation_model": os.getenv("LOCAL_GENERATION_MODEL", "mistral"),
    }


def _setenv_if_value(name: str, value: Optional[str]) -> None:
    text = (value or "").strip()
    if text:
        os.environ[name] = text
