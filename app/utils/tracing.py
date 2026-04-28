"""Lightweight trace append helpers."""

import time
from typing import Any, Dict

from app.core.stability_patch import safe_get, with_updates
from app.state import AgentState


def trace(state: AgentState, node: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
    log = list(safe_get(state, "trace_log", []) or [])
    log.append({"node": node, "ts": round(time.time(), 3), "data": data or {}})
    return with_updates(state, trace_log=log)
