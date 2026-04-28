"""Rule-based intent router for offline-first execution."""

import logging

from app.core.stability_patch import safe_get, with_updates

logger = logging.getLogger(__name__)

VALID_LABELS = {"chat", "rag", "compute", "unclear"}


def _rule_based_intent(query: str) -> str:
    lower = query.lower()

    if any(word in lower for word in ["hi", "hello", "how are you", "who are you"]):
        return "chat"

    if any(word in lower for word in ["calculate", "total", "cost", "percent", "percentage"]):
        return "compute"

    if any(
        word in lower
        for word in [
            "policy",
            "allowance",
            "rule",
            "eligibility",
            "reimbursement",
            "fuel",
            "travel",
            "leave",
            "security",
        ]
    ):
        return "rag"

    return "unclear"


def _llm_intent(query: str) -> str:
    return "unclear"


def classify_intent(query: str) -> str:
    return _rule_based_intent(query)


def run(state):
    query = (safe_get(state, "query", "") or "").strip()
    trace_log = list(safe_get(state, "trace_log", []) or [])

    try:
        intent = classify_intent(query) if query else "unclear"
    except Exception as exc:
        logger.exception("Router crashed, falling back to RAG")
        trace_log.append(f"router fallback: {exc}")
        intent = "rag"

    if intent == "unclear":
        intent = "rag"
        trace_log.append("router fallback: unclear intent routed to rag")

    route = "retrieval" if intent == "rag" else intent
    return with_updates(state, intent=intent, route=route, trace_log=trace_log)
