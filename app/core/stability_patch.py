"""
Stability helpers for safe graph invocation and mixed state access.
"""

import logging
from typing import Any, Dict, Optional

from app.state import AgentState, state_to_dict, to_state

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

        return {
            "answer": safe_get(result, "answer", "No response generated"),
            "route": safe_get(result, "route", "unknown"),
            "intent": safe_get(result, "intent", "unknown"),
            "sources": safe_get(result, "sources", []),
            "trace_log": safe_get(result, "trace_log", []),
            "error": safe_get(result, "error", ""),
        }

    except KeyError as e:
        logger.error("Missing required state key: %s", e, exc_info=True)
        return _error_result(f"Configuration error: {str(e)}", e)

    except Exception as e:
        logger.error("Graph execution failed: %s", e, exc_info=True)
        return _error_result("System error. Please try again.", e)


def _error_result(message: str, error: Exception) -> Dict[str, Any]:
    return {
        "answer": message,
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
