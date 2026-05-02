"""Model helpers for AEGIS generation."""

import logging
import re
import urllib.request
from types import SimpleNamespace
from typing import Optional

from app.core.settings import get_settings

logger = logging.getLogger(__name__)

_OFFLINE_PROVIDERS = {"extractive", "offline", "local", "none", "false", "0"}
_OPENAI_PROVIDERS = {"openai", "gpt", "gpt-4o-mini"}
_OLLAMA_PROVIDERS = {"ollama", "auto"}


def get_embed_model() -> str:
    """Return the configured embedding model name."""
    settings = get_settings()
    if settings.rag_embeddings_provider.strip().lower() == "openai":
        return settings.openai_embedding_model
    return settings.local_embed_model


class LocalPolicyModel:
    """Small extractive fallback used when no hosted or local LLM is available."""

    def invoke(self, messages):
        prompt = _messages_to_prompt(messages)
        content = _extractive_answer(prompt)
        return SimpleNamespace(content=content)


class OpenAIPolicyModel:
    """OpenAI Responses API adapter returning a LangChain-like response object."""

    def __init__(self, model: str, api_key: str, temperature: float = 0.1, max_tokens: Optional[int] = None):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

    def invoke(self, messages):
        from openai import OpenAI

        prompt = _messages_to_prompt(messages)
        client = OpenAI(api_key=self.api_key)
        kwargs = {
            "model": self.model,
            "input": prompt,
        }
        if self.max_tokens:
            kwargs["max_output_tokens"] = self.max_tokens
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        response = client.responses.create(**kwargs)
        content = _openai_response_text(response)
        return SimpleNamespace(content=content)


def _messages_to_prompt(messages) -> str:
    if isinstance(messages, str):
        return messages

    parts = []
    for message in messages or []:
        if isinstance(message, dict):
            role = message.get("role", "user")
            content = message.get("content", "")
            parts.append(f"{role}: {content}")
        else:
            parts.append(str(getattr(message, "content", message)))
    return "\n\n".join(parts)


def _openai_response_text(response) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return text

    output = getattr(response, "output", None) or []
    for item in output:
        content = getattr(item, "content", None) or []
        for part in content:
            part_text = getattr(part, "text", None)
            if part_text:
                return part_text

    return str(response)


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


def _ollama_base_url() -> str:
    settings = get_settings()
    base_url = settings.ollama_host or settings.ollama_base_url or "http://localhost:11434"
    base_url = base_url.strip().rstrip("/")
    if "://" not in base_url:
        base_url = f"http://{base_url}"
    return base_url


def _ollama_server_available(base_url: str) -> bool:
    settings = get_settings()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/tags",
        headers={"User-Agent": "aegis-rag-system"},
    )
    try:
        with urllib.request.urlopen(request, timeout=settings.ollama_health_timeout) as response:
            return 200 <= int(response.status) < 500
    except Exception as exc:
        logger.warning("Ollama is not reachable at %s; using extractive fallback: %s", base_url, exc)
        return False


def get_llm(model_override: Optional[str] = None, temperature: Optional[float] = None, max_tokens: Optional[int] = None):
    """Return the configured model with safe fallbacks when credentials are absent."""
    settings = get_settings()
    provider = settings.active_llm_provider

    if provider in _OFFLINE_PROVIDERS:
        logger.info("Using extractive local model provider")
        return LocalPolicyModel()

    if provider in _OPENAI_PROVIDERS:
        if not settings.openai_key:
            logger.warning("OPENAI_API_KEY is not set; using extractive fallback")
            return LocalPolicyModel()
        return OpenAIPolicyModel(
            model=model_override or settings.openai_model,
            api_key=settings.openai_key,
            temperature=settings.openai_temperature if temperature is None else temperature,
            max_tokens=max_tokens or settings.openai_max_output_tokens,
        )

    if provider in _OLLAMA_PROVIDERS:
        base_url = _ollama_base_url()
        if not _ollama_server_available(base_url):
            return LocalPolicyModel()

        model = model_override or settings.ollama_model
        try:
            from langchain_community.llms import Ollama

            kwargs = {"model": model, "temperature": temperature or 0.0, "base_url": base_url}
            if max_tokens is not None:
                kwargs["num_predict"] = max_tokens
            return Ollama(**kwargs)
        except Exception as exc:
            logger.warning("Local LLM unavailable, using extractive fallback: %s", exc)
            return LocalPolicyModel()

    logger.warning("Unknown LLM_PROVIDER=%s; using extractive fallback", provider)
    return LocalPolicyModel()


def invoke_llm(messages: list, model_override: Optional[str] = None, temperature: float = 0):
    """Invoke the configured generation stack."""
    llm = get_llm(model_override=model_override, temperature=temperature)
    try:
        return llm.invoke(messages)
    except Exception as exc:
        logger.warning("Model call failed, using extractive fallback: %s", exc)
        return LocalPolicyModel().invoke(messages)
