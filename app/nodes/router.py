"""
Hybrid intent router.

The router accepts either plain dict state or AgentState objects. On ambiguity or
classifier failure it falls back to RAG instead of crashing the graph.
"""

import logging

from app.core.models import get_llm
from app.core.stability_patch import safe_get, with_updates

logger = logging.getLogger(__name__)

VALID_LABELS = {"chat", "rag", "compute", "unclear"}


def _rule_based_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["hi", "hello", "how are you", "who are you"]):
        return "chat"

    if any(x in q for x in ["calculate", "total", "cost", "percent", "percentage"]):
        return "compute"

    if any(x in q for x in ["policy", "allowance", "rule", "eligibility", "reimbursement"]):
        return "rag"

    return "unclear"


def _llm_intent(query: str) -> str:
    try:
        llm = get_llm(model_override="gpt-4o-mini", temperature=0)
        prompt = f"""
You are a strict classifier.

Return ONLY one word from:
chat, rag, compute, unclear

Query: {query}
"""
        res = llm.invoke(prompt)
        out = getattr(res, "content", "").strip().lower()
    except Exception as exc:
        logger.warning("Router LLM fallback failed: %s", exc)
        return "unclear"

    if out not in VALID_LABELS:
        return "unclear"

    return out


def classify_intent(query: str) -> str:
    rule_intent = _rule_based_intent(query)
    if rule_intent != "unclear":
        return rule_intent
    return _llm_intent(query)


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
