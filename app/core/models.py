"""Model helpers for AEGIS generation."""

import logging
import re
from types import SimpleNamespace
from typing import Optional

from app.core.llm_decision_manager import LOCAL_PROVIDERS, messages_to_prompt, select_local_llm
from app.core.settings import get_settings

logger = logging.getLogger(__name__)

_OFFLINE_PROVIDERS = {"extractive", "offline", "local", "none", "false", "0"}
_LOCAL_PROVIDERS = LOCAL_PROVIDERS | {"local_auto"}


def get_embed_model() -> str:
    """Return the configured embedding model name."""
    settings = get_settings()
    provider = settings.active_embeddings_provider
    if provider == "openai":
        return settings.openai_embedding_model
    if provider in {"google", "gemini", "google-gemini"}:
        return settings.google_embedding_model
    return settings.local_embed_model


class LocalPolicyModel:
    """Small extractive fallback used when no local LLM runtime is available."""

    decision = {"provider": "extractive", "model": "extractive"}

    def invoke(self, messages):
        prompt = messages_to_prompt(messages)
        content = _extractive_answer(prompt)
        return SimpleNamespace(content=content, model_provider="extractive", model_name="extractive", model_decision=self.decision)


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


def get_llm(
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    node: str = "generator",
):
    """Return a local-only model adapter with safe extractive fallback."""
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

    if provider in _GOOGLE_PROVIDERS:
        if not settings.google_key:
            logger.warning("GEMINI_API_KEY or GOOGLE_API_KEY is not set; using extractive fallback")
            return LocalPolicyModel()
        return GooglePolicyModel(
            model=model_override or settings.google_model,
            api_key=settings.google_key,
            temperature=settings.google_temperature if temperature is None else temperature,
            max_tokens=max_tokens or settings.google_max_output_tokens,
        )

    if provider in _OLLAMA_PROVIDERS:
        base_url = _ollama_base_url()
        if not _ollama_server_available(base_url):
            return LocalPolicyModel()

        model = model_override or settings.ollama_model
        try:
            from langchain_community.llms import Ollama

            kwargs = {"model": model, "temperature": temperature if temperature is not None else 0.0, "base_url": base_url}
            if max_tokens is not None:
                kwargs["num_predict"] = max_tokens
            return Ollama(**kwargs)
        except Exception as exc:
            logger.warning("Local LLM unavailable, using extractive fallback: %s", exc)
            return LocalPolicyModel()

    logger.warning("Unknown LLM_PROVIDER=%s; using extractive fallback", provider)
    return LocalPolicyModel()


    llm, decision = select_local_llm(
        node=node,
        provider=provider,
        model_override=model_override,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if llm is not None:
        return llm

    fallback = LocalPolicyModel()
    fallback.decision = decision.as_dict()
    logger.warning("No configured local LLM runtime is available; using extractive fallback: %s", fallback.decision)
    return fallback


def invoke_llm(
    messages: list,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    node: str = "generator",
):
    """Invoke the local generation stack selected for the current node."""
    llm = get_llm(model_override=model_override, temperature=temperature, node=node)
    try:
        response = llm.invoke(messages)
    except Exception as exc:
        logger.warning("Model call failed, using extractive fallback: %s", exc)
        fallback = LocalPolicyModel()
        response = fallback.invoke(messages)

    if not getattr(response, "model_decision", None):
        response.model_decision = getattr(llm, "decision", {"provider": "extractive", "model": "extractive"})
    return response
