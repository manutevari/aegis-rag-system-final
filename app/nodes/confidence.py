# app/nodes/confidence.py
"""
Confidence Scoring — computes a 0–1 score for answer quality
"""

import re
from app.state import AgentState
from app.utils.tracing import trace

# simple weights (tune later)
W_RETRIEVAL = 0.25
W_SOURCES   = 0.20
W_COMPLETENESS = 0.25
W_POLICY_SAFETY = 0.15
W_VERIFIER  = 0.15

def _has_sources(answer: str) -> float:
    return 1.0 if "Source:" in answer else 0.0

def _completeness(answer: str, query: str) -> float:
    # naive coverage: fraction of query keywords appearing in answer
    q_tokens = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 3]
    if not q_tokens:
        return 0.7
    hits = sum(1 for t in q_tokens if t in answer.lower())
    return min(1.0, hits / max(1, len(q_tokens)))

def _policy_safety(answer: str, context: str) -> float:
    # penalize numbers not present in context
    nums_ans = set(re.findall(r"\d[\d,\.]*", answer))
    nums_ctx = set(re.findall(r"\d[\d,\.]*", context))
    if not nums_ans:
        return 0.8
    ok = sum(1 for n in nums_ans if n in nums_ctx)
    return ok / max(1, len(nums_ans))

def run(state: AgentState) -> AgentState:
    answer   = state.get("answer", "")
    query    = state.get("query", "")
    context  = state.get("context", "")
    verified = bool(state.get("verified"))

    retrieval_score = float(state.get("retrieval_score", 0.7))  # fallback

    s_sources = _has_sources(answer)
    s_comp    = _completeness(answer, query)
    s_policy  = _policy_safety(answer, context)
    s_ver     = 1.0 if verified else 0.0

    confidence = (
        W_RETRIEVAL * retrieval_score +
        W_SOURCES   * s_sources +
        W_COMPLETENESS * s_comp +
        W_POLICY_SAFETY * s_policy +
        W_VERIFIER  * s_ver
    )

    confidence = max(0.0, min(1.0, confidence))

    state["confidence"] = confidence

    return trace(
        state,
        node="confidence",
        data={
            "confidence": round(confidence, 3),
            "components": {
                "retrieval": retrieval_score,
                "sources": s_sources,
                "completeness": round(s_comp, 3),
                "policy_safety": round(s_policy, 3),
                "verifier": s_ver
            }
        }
    )
