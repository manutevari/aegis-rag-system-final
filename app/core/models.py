"""Local model helpers for offline-first RAG."""

import logging
import os
import re
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger(__name__)


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


def get_llm(model_override: Optional[str] = None, temperature: float = 0.0, max_tokens: Optional[int] = None):
    """Return a local Ollama model, with an extractive fallback."""
    model = model_override or os.getenv("OLLAMA_MODEL", "llama3")
    try:
        from langchain_community.llms import Ollama

        return Ollama(model=model, temperature=temperature)
    except Exception as exc:
        logger.warning("Local LLM unavailable, using extractive fallback: %s", exc)
        return LocalPolicyModel()


def invoke_llm(messages: list, model_override: Optional[str] = None, temperature: float = 0):
    """Invoke the local model stack."""
    llm = get_llm(model_override=model_override, temperature=temperature)
    try:
        return llm.invoke(messages)
    except Exception as exc:
        logger.warning("Local LLM call failed, using extractive fallback: %s", exc)
        return LocalPolicyModel().invoke(messages)
