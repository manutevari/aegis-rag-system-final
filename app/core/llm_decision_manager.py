"""Local-only LLM decision manager for node-specific orchestration."""

import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from app.core.settings import AppSettings, get_settings

logger = logging.getLogger(__name__)

LOCAL_PROVIDERS = {"local_auto", "auto", "ollama", "llama_cpp", "llama.cpp", "mistral_local", "extractive"}
_NODE_ROLES = {
    "planner": "orchestration",
    "router": "orchestration",
    "chat": "orchestration",
    "summarizer": "summary",
    "generator": "generation",
}


@dataclass
class LocalModelDecision:
    node: str
    role: str
    provider: str
    model: str = ""
    base_url: str = ""
    skipped: List[Dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "skipped": self.skipped,
        }


class LocalHTTPModel:
    provider = "local"

    def __init__(self, model: str, base_url: str, temperature: float = 0.0, max_tokens: Optional[int] = None):
        self.model = model
        self.base_url = _clean_base_url(base_url)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.decision: Dict[str, Any] = {}

    def availability(self, timeout: float) -> Tuple[bool, str]:
        raise NotImplementedError

    def invoke(self, messages):
        raise NotImplementedError


class OllamaModel(LocalHTTPModel):
    provider = "ollama"

    def availability(self, timeout: float) -> Tuple[bool, str]:
        try:
            data = _json_request(f"{self.base_url}/api/tags", timeout=timeout)
        except Exception as exc:
            return False, f"Ollama not reachable: {exc}"

        names = []
        for item in data.get("models", []) or []:
            name = item.get("name") if isinstance(item, dict) else str(item)
            if name:
                names.append(name)

        if names and not _model_name_matches(self.model, names):
            return False, f"Ollama model {self.model} not found"
        return True, "ok"

    def invoke(self, messages):
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": messages_to_prompt(messages),
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens

        data = _json_request(f"{self.base_url}/api/generate", payload=payload, timeout=120)
        return SimpleNamespace(
            content=data.get("response", ""),
            model_provider=self.provider,
            model_name=self.model,
            model_decision=self.decision,
        )


class OpenAICompatibleLocalModel(LocalHTTPModel):
    provider = "openai_compatible_local"

    def availability(self, timeout: float) -> Tuple[bool, str]:
        urls = [f"{self.base_url}/health", f"{self.base_url}/models"]
        last_error = "not checked"
        for url in urls:
            try:
                _json_request(url, timeout=timeout)
                return True, "ok"
            except Exception as exc:
                last_error = str(exc)
        return False, f"{self.provider} not reachable: {last_error}"

    def invoke(self, messages):
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages_to_chat(messages),
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens

        data = _json_request(f"{self.base_url}/chat/completions", payload=payload, timeout=120)
        content = ""
        choices = data.get("choices", []) or []
        if choices:
            message = choices[0].get("message", {}) or {}
            content = message.get("content", "")
        return SimpleNamespace(
            content=content,
            model_provider=self.provider,
            model_name=self.model,
            model_decision=self.decision,
        )


class LlamaCppModel(OpenAICompatibleLocalModel):
    provider = "llama_cpp"


class MistralLocalModel(OpenAICompatibleLocalModel):
    provider = "mistral_local"


class LocalLLMDecisionManager:
    """Chooses which local runtime should serve a specific node/tool role."""

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or get_settings()

    def select(
        self,
        node: str = "generator",
        provider: Optional[str] = None,
        model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[Optional[LocalHTTPModel], LocalModelDecision]:
        role = _role_for_node(node)
        requested_provider = _normalize_provider(provider or self.settings.active_llm_provider)
        skipped: List[Dict[str, str]] = []

        if requested_provider == "extractive":
            return None, LocalModelDecision(node=node, role=role, provider="extractive", skipped=[])

        for candidate_provider in self._provider_order(requested_provider):
            adapter = self._adapter_for(candidate_provider, role, model_override, temperature, max_tokens)
            ok, reason = adapter.availability(timeout=self.settings.local_llm_health_timeout)
            if ok:
                decision = LocalModelDecision(
                    node=node,
                    role=role,
                    provider=adapter.provider,
                    model=adapter.model,
                    base_url=adapter.base_url,
                    skipped=skipped,
                )
                adapter.decision = decision.as_dict()
                logger.info("Selected local LLM provider=%s model=%s node=%s", adapter.provider, adapter.model, node)
                return adapter, decision
            skipped.append({"provider": adapter.provider, "model": adapter.model, "reason": reason})

        return None, LocalModelDecision(node=node, role=role, provider="extractive", skipped=skipped)

    def _provider_order(self, requested_provider: str) -> List[str]:
        if requested_provider == "local_auto":
            return [
                _normalize_provider(item)
                for item in (self.settings.local_llm_order or "ollama,llama_cpp,mistral_local").split(",")
                if _normalize_provider(item) != "extractive"
            ]
        return [requested_provider]

    def _adapter_for(
        self,
        provider: str,
        role: str,
        model_override: Optional[str],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> LocalHTTPModel:
        model = (model_override or self._model_for(provider, role)).strip()
        temp = self.settings.local_llm_temperature if temperature is None else temperature

        if provider == "ollama":
            return OllamaModel(model=model, base_url=self.settings.ollama_base_url, temperature=temp, max_tokens=max_tokens)
        if provider == "llama_cpp":
            return LlamaCppModel(model=model, base_url=self.settings.llama_cpp_base_url, temperature=temp, max_tokens=max_tokens)
        if provider == "mistral_local":
            return MistralLocalModel(model=model, base_url=self.settings.mistral_local_base_url, temperature=temp, max_tokens=max_tokens)
        return OllamaModel(model=model, base_url=self.settings.ollama_base_url, temperature=temp, max_tokens=max_tokens)

    def _model_for(self, provider: str, role: str) -> str:
        if provider == "llama_cpp":
            return self.settings.llama_cpp_model
        if provider == "mistral_local":
            return self.settings.mistral_local_model
        if role == "orchestration":
            return self.settings.local_orchestration_model or self.settings.ollama_model
        if role == "summary":
            return self.settings.local_summary_model or self.settings.local_generation_model
        return self.settings.local_generation_model or self.settings.ollama_model


def select_local_llm(
    node: str = "generator",
    provider: Optional[str] = None,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    return LocalLLMDecisionManager().select(
        node=node,
        provider=provider,
        model_override=model_override,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def messages_to_prompt(messages) -> str:
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


def messages_to_chat(messages) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]

    chat_messages = []
    for message in messages or []:
        if isinstance(message, dict):
            role = message.get("role") or "user"
            content = message.get("content") or ""
        else:
            role = "user"
            content = str(getattr(message, "content", message))
        if role not in {"system", "user", "assistant"}:
            role = "user"
        chat_messages.append({"role": role, "content": str(content)})
    return chat_messages or [{"role": "user", "content": ""}]


def _role_for_node(node: str) -> str:
    return _NODE_ROLES.get((node or "generator").strip().lower(), "generation")


def _normalize_provider(provider: str) -> str:
    value = (provider or "local_auto").strip().lower().replace("-", "_")
    if value in {"auto", "local", "local_auto"}:
        return "local_auto"
    if value in {"llama.cpp", "llamacpp", "llama_cpp"}:
        return "llama_cpp"
    if value in {"mistral", "mistral_local", "local_mistral"}:
        return "mistral_local"
    if value == "ollama":
        return "ollama"
    if value in {"extractive", "offline", "none", "false", "0"}:
        return "extractive"
    return "local_auto"


def _clean_base_url(base_url: str) -> str:
    value = (base_url or "").strip().rstrip("/")
    if not value:
        return "http://localhost:11434"
    if "://" not in value:
        value = f"http://{value}"
    if value.endswith("/v1"):
        return value
    return value


def _json_request(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 2.0) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "aegis-rag-system"},
        method="POST" if payload is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:200]}") from exc


def _model_name_matches(requested: str, available: List[str]) -> bool:
    wanted = (requested or "").split(":", 1)[0]
    for name in available:
        if name == requested:
            return True
        if name.split(":", 1)[0] == wanted:
            return True
    return False
