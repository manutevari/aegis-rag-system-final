# app/nodes/retry_controller.py
"""
Retry Controller — bounded loop for LLM stability
- Escalates model on failure
- Prevents infinite loops
- Routes to HITL when exhausted
"""

import logging
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

MAX_RETRIES = 2

# simple tier ladder (customize as needed)
MODEL_TIERS = [
    "gpt-4o-mini",  # fast/cheap
    "gpt-5-mini",   # stronger
    "gpt-5-nano",   # alt tier if you prefer; reorder if needed
]

def _next_model(current: str) -> str:
    try:
        idx = MODEL_TIERS.index(current)
        return MODEL_TIERS[min(idx + 1, len(MODEL_TIERS) - 1)]
    except ValueError:
        return MODEL_TIERS[0]

def run(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    verified = bool(state.get("verified"))
    retries = int(state.get("retries", 0))
    current_model = state.get("model") or MODEL_TIERS[0]

    # treat explicit generation errors as failure
    has_error = (
        state.get("error") is not None
        or "Generation error" in answer
        or "⚠️" in answer
    )

    # success path
    if verified and not has_error:
        state["route"] = "trace"
        return trace(state, node="retry_controller", data={
            "decision": "pass",
            "retries": retries,
            "model": current_model
        })

    # failure path
    if retries >= MAX_RETRIES:
        state["route"] = "hitl"
        return trace(state, node="retry_controller", data={
            "decision": "exhausted_to_hitl",
            "retries": retries,
            "model": current_model
        })

    # retry with escalation
    next_model = _next_model(current_model)
    state["retries"] = retries + 1
    state["model"] = next_model
    state["route"] = "generate"  # loop back

    logger.warning(
        f"Retry #{state['retries']} — escalating model: {current_model} → {next_model}"
    )

    return trace(state, node="retry_controller", data={
        "decision": "retry",
        "retries": state["retries"],
        "model": next_model
    })
