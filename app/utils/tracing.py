"""Lightweight immutable trace append."""
import time
from typing import Any, Dict
from app.state import AgentState

def trace(state: AgentState, node: str, data: Dict[str, Any] = None) -> AgentState:
    log = list(state.get("trace_log") or [])
    log.append({"node": node, "ts": round(time.time(), 3), "data": data or {}})
    return {**state, "trace_log": log}
