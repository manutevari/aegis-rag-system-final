"""Local model helpers for offline-first RAG."""

import logging
import os
import re
import urllib.request
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)

_OFFLINE_PROVIDERS = {"extractive", "offline", "local", "none", "false", "0"}
_OLLAMA_PROVIDERS = {"ollama", "auto"}


def get_embed_model() -> str:
    """Return the local embedding model name."""
    return os.getenv("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class LocalPolicyModel:
    """Small extractive fallback used when no local LLM server is available."""

    def invoke(self, messages):
        prompt = _messages_to_prompt(messages)
        content = _extractive_answer(prompt)
        return SimpleNamespace(content=content)


def _messages_to_prompt(messages) -> str:
    if isinstance(messages, str):
        return messages

    parts = []
    for message in messages or []:
        if isinstance(message, dict):
            parts.append(str(message.get("content", "")))
        else:
            parts.append(str(getattr(message, "content", message)))
    return "\n\n".join(parts)


def _extractive_answer(prompt: str) -> str:
    context_match = re.search(r"POLICY CONTEXT:\s*(.*?)\s*QUESTION:", prompt, flags=re.I | re.S)
    question_match = re.search(r"QUESTION:\s*(.*)", prompt, flags=re.I | re.S)

    context = context_match.group(1).strip() if context_match else prompt
    question = question_match.group(1).strip() if question_match else ""
    terms = {term for term in re.findall(r"[a-z0-9]+", question.lower()) if len(term) > 2}

    sentences = re.split(r"(?<=[.!?])\s+|\n+", context)
    ranked = []
    for sentence in sentences:
        clean = sentence.strip()
        if not clean:
            continue
        words = set(re.findall(r"[a-z0-9]+", clean.lower()))
        score = len(words & terms)
        if score:
            ranked.append((score, clean))

    selected = [text for _, text in sorted(ranked, reverse=True)[:5]]
    if not selected:
        selected = [line.strip() for line in context.splitlines() if line.strip()][:5]

    if not selected:
        return "This is not covered in the available policy data."

    return "\n".join(f"- {line}" for line in selected)


def _llm_provider() -> str:
    return (os.getenv("LLM_PROVIDER") or os.getenv("MODEL_PROVIDER") or "extractive").strip().lower()


def _ollama_base_url() -> str:
    base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
    base_url = base_url.strip().rstrip("/")
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    return base_url


def _ollama_server_available(base_url: str) -> bool:
    timeout = float(os.getenv("OLLAMA_HEALTH_TIMEOUT", "0.5"))
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"User-Agent": "aegis-rag-system"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except Exception as exc:
        logger.warning("Ollama is not reachable at %s; using extractive fallback: %s", base_url, exc)
        return False


def get_llm(model_override: Optional[str] = None, temperature: float = 0.0, max_tokens: Optional[int] = None):
    """Return the configured local model without requiring Ollama by default."""
    provider = _llm_provider()
    if provider in _OFFLINE_PROVIDERS:
        logger.info("Using extractive local model provider")
        return LocalPolicyModel()

    if provider not in _OLLAMA_PROVIDERS:
        logger.warning("Unknown LLM_PROVIDER=%s; using extractive fallback", provider)
        return LocalPolicyModel()

    base_url = _ollama_base_url()
    if not _ollama_server_available(base_url):
        return LocalPolicyModel()

    model = model_override or os.getenv("OLLAMA_MODEL", "llama3")
    try:
        from langchain_community.llms import Ollama

        kwargs = {"model": model, "temperature": temperature, "base_url": base_url}
        if max_tokens is not None:
            kwargs["num_predict"] = max_tokens
        return Ollama(**kwargs)
    except Exception as exc:
        logger.warning("Local LLM unavailable, using extractive fallback: %s", exc)
        return LocalPolicyModel()


def invoke_llm(messages: list, model_override: Optional[str] = None, temperature: float = 0):
    """Invoke the configured local model stack."""
    llm = get_llm(model_override=model_override, temperature=temperature)
    try:
        return llm.invoke(messages)
    except Exception as exc:
        logger.warning("Local LLM call failed, using extractive fallback: %s", exc)
        return LocalPolicyModel().invoke(messages)
