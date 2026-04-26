"""
Retry Controller — bounded loop for LLM stability (confidence-aware)

- Uses confidence scoring
- Escalates model only when needed
- Prevents infinite loops
- Routes to HITL when exhausted
"""

import logging
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

# ==============================
# 🔹 Config
# ==============================

MAX_RETRIES = 2
TH_ACCEPT = 0.75
TH_RETRY  = 0.50

MODEL_TIERS = [
    "gpt-4o-mini",  # fast
    "gpt-5-mini",   # stronger
    "gpt-5-nano",   # alternative/escalation
]


# ==============================
# 🔹 Model Escalation
# ==============================

def _next_model(current: str) -> str:
    try:
        idx = MODEL_TIERS.index(current)
        return MODEL_TIERS[min(idx + 1, len(MODEL_TIERS) - 1)]
    except ValueError:
        return MODEL_TIERS[0]


# ==============================
# 🔹 Main Controller
# ==============================

def run(state: AgentState) -> AgentState:
    answer = (state.get("answer") or "").strip()
    verified = bool(state.get("verified"))
    retries = int(state.get("retries", 0))
    confidence = float(state.get("confidence", 0.0))
    current_model = state.get("model") or MODEL_TIERS[0]

    # 🔴 Detect hard failure
    has_error = (
        state.get("error") is not None
        or "Generation error" in answer
        or "⚠️" in answer
    )

    # ==============================
    # ✅ ACCEPT (High confidence)
    # ==============================
    if not has_error and verified and confidence >= TH_ACCEPT:
        state["route"] = "trace"

        return trace(state, node="retry_controller", data={
            "decision": "accept",
            "confidence": round(confidence, 3),
            "retries": retries,
            "model": current_model
        })

    # ==============================
    # 🔁 RETRY (Smart logic)
    # ==============================
    if retries < MAX_RETRIES and (has_error or confidence < TH_ACCEPT):

        # escalate only if very low confidence
        if confidence < TH_RETRY:
            next_model = _next_model(current_model)
        else:
            next_model = current_model

        state["retries"] = retries + 1
        state["model"] = next_model
        state["route"] = "generate"

        logger.warning(
            f"Retry #{state['retries']} | conf={confidence:.2f} | {current_model} → {next_model}"
        )

        return trace(state, node="retry_controller", data={
            "decision": "retry",
            "confidence": round(confidence, 3),
            "retries": state["retries"],
            "model": next_model,
            "reason": "error" if has_error else "low_confidence"
        })

    # ==============================
    # 🧑‍💼 ESCALATE TO HITL
    # ==============================
    state["route"] = "hitl"

    return trace(state, node="retry_controller", data={
        "decision": "hitl",
        "confidence": round(confidence, 3),
        "retries": retries,
        "model": current_model
    })
