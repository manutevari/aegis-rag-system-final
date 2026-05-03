"""Runtime configuration helpers for Streamlit local model controls."""

import os
from typing import Dict, Optional

from app.core.settings import get_settings

_PROVIDER_ALIASES = {
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
