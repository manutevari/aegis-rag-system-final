"""
Stability helpers for safe graph invocation and mixed state access.
"""

import logging
import os
from typing import Any, Dict, Optional

from app.state import AgentState, state_to_dict, to_state
from app.tools.tavily_search import TAVILY_ENV, tavily_context_block, tavily_search

logger = logging.getLogger(__name__)


class StateAdapter:
    def __init__(self, state: Any):
        self._state = state

    def get(self, key: str, default: Any = None) -> Any:
        return safe_get(self._state, key, default)

    def set(self, key: str, value: Any) -> Any:
        return safe_set(self._state, key, value)

    def to_dict(self) -> Dict[str, Any]:
        return as_dict(self._state)


def safe_get(state: Any, key: str, default: Any = None) -> Any:
    if state is None:
        return default
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def safe_set(state: Any, key: str, value: Any) -> Any:
    if isinstance(state, dict):
        state[key] = value
    else:
        setattr(state, key, value)
    return state


def as_dict(state: Any) -> Dict[str, Any]:
    if isinstance(state, AgentState):
        return state_to_dict(state)
    if hasattr(state, "model_dump"):
        return state.model_dump()
    return dict(state or {})


def with_updates(state: Any, **updates: Any) -> Dict[str, Any]:
    data = as_dict(state)
    data.update(updates)
    return data


def safe_invoke(graph, initial_state: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke a graph and return a UI-friendly result payload."""
    try:
        sanitized_state = _sanitize_state(initial_state)
        result = graph.invoke(sanitized_state)
        payload = {
            "answer": safe_get(result, "answer", "No response generated"),
            "context": safe_get(result, "context", ""),
            "documents": safe_get(result, "documents", []),
            "route": safe_get(result, "route", "unknown"),
            "intent": safe_get(result, "intent", "unknown"),
            "sources": safe_get(result, "sources", []),
            "trace_log": safe_get(result, "trace_log", []),
            "error": safe_get(result, "error", ""),
        }

        return _augment_with_tavily(payload, sanitized_state)

    except KeyError as e:
        logger.error("Missing required state key: %s", e, exc_info=True)
        return _error_result(f"Configuration error: {str(e)}", e)

    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        return _error_result("System error. Please try again.", e)


def _error_result(message: str, error: Exception) -> Dict[str, Any]:
    return {
        "answer": message,
        "context": "",
        "documents": [],
        "route": "error",
        "intent": "error",
        "sources": [],
        "trace_log": [str(error)],
        "error": str(error),
    }


def _sanitize_state(state: Any) -> Dict[str, Any]:
    raw = StateAdapter(state)
    sanitized = {
        "query": (raw.get("query", "") or "").strip(),
        "history": raw.get("history") or [],
        "memory_context": raw.get("memory_context") or "",
        "trace_log": raw.get("trace_log") or [],
        "employee_grade": raw.get("employee_grade") or "L3",
        "route": raw.get("route") or "retrieval",
        "intent": raw.get("intent") or "rag",
    }

    if not isinstance(sanitized["history"], list):
        sanitized["history"] = []

    if not isinstance(sanitized["trace_log"], list):
        sanitized["trace_log"] = []

    if not isinstance(sanitized["memory_context"], str):
        sanitized["memory_context"] = ""

    if not sanitized["query"]:
        raise ValueError("Query cannot be empty")

    return to_state(sanitized).to_dict()


def _session_value(key: str, default: Any = None) -> Any:
    try:
        import streamlit as st

        return st.session_state.get(key, default)
    except Exception:
        return default


def _secret_or_env(names) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    try:
        import streamlit as st

        for name in names:
            value = st.secrets.get(name, "")
            if value:
                return value
    except Exception:
        pass
    return ""


def _tavily_controls() -> Dict[str, Any]:
    api_key = _session_value("tavily_api_key") or _secret_or_env(TAVILY_ENV)
    enabled_default = bool(api_key)
    enabled = bool(_session_value("tavily_enabled", enabled_default))
    try:
        max_results = int(_session_value("tavily_max_results", 3) or 3)
    except Exception:
        max_results = 3
    return {
        "enabled": enabled,
        "api_key": api_key,
        "search_depth": _session_value("tavily_search_depth", "basic") or "basic",
        "max_results": max(1, min(max_results, 10)),
    }


def _run_tavily(query: str) -> Dict[str, Any]:
    controls = _tavily_controls()
    if not controls["enabled"]:
        return {
            "enabled": False,
            "status": "disabled",
            "answer": "",
            "results": [],
            "sources": [],
            "error": "",
        }
    if not controls["api_key"]:
        return {
            "enabled": True,
            "status": "missing_key",
            "answer": "",
            "results": [],
            "sources": [],
            "error": "Tavily API key is not configured",
        }
    try:
        return tavily_search(
            query,
            api_key=controls["api_key"],
            search_depth=controls["search_depth"],
            max_results=controls["max_results"],
            include_answer="basic",
        )
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc, exc_info=True)
        return {
            "enabled": True,
            "status": "error",
            "answer": "",
            "results": [],
            "sources": [],
            "error": str(exc),
        }


def _augment_with_tavily(payload: Dict[str, Any], sanitized_state: Dict[str, Any]) -> Dict[str, Any]:
    tavily_state = _run_tavily(sanitized_state.get("query", ""))
    payload["tavily"] = tavily_state
    payload["trace_log"] = list(payload.get("trace_log") or [])

    if tavily_state.get("status") != "disabled":
        payload["trace_log"].append(
            {
                "node": "tavily_search",
                "data": {
                    "enabled": tavily_state.get("enabled", False),
                    "status": tavily_state.get("status", "unknown"),
                    "results": len(tavily_state.get("results", [])),
                    "sources": tavily_state.get("sources", []),
                    "error": tavily_state.get("error", ""),
                },
            }
        )

    if tavily_state.get("enabled") and tavily_state.get("results"):
        supplemental = (
            "\n\n<tavily_context>\n"
            "Supplementary web-search evidence from Tavily. Use only when it supports "
            "the policy context, and never override corporate policy text with web data.\n"
            f"{tavily_context_block(tavily_state)}\n"
            "</tavily_context>"
        )
        payload["context"] = f"{payload.get('context', '')}{supplemental}"
        sources = list(payload.get("sources") or [])
        for source in tavily_state.get("sources", []):
            if source and source not in sources:
                sources.append(source)
        payload["sources"] = sources

    return payload
