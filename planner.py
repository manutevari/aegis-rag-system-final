"""
Planner Node — Routes query to the correct tool path.

Grade detection:  Extracts employee grade from query or history so downstream
                  SQL/compute nodes can filter policy tables precisely.

Routing logic:
  "sql"       — structured entitlement lookups (rates, limits, budgets)
  "retrieval" — policy procedure / process questions
  "compute"   — pure arithmetic on already-known figures
  "direct"    — conversational / clarification
"""

import json
import logging
import os
import re

from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

# Grade normalisation map
_GRADE_MAP = {
    "l1": "L1", "l2": "L2", "l3": "L3", "l4": "L4",
    "l5": "L5", "l6": "L6", "l7": "L7",
    "vp": "VP", "svp": "SVP", "cxo": "CXO",
    "manager": "L5", "senior manager": "L6",
    "analyst": "L2", "associate": "L3", "consultant": "L4",
    "director": "L7", "executive director": "VP",
}

_SQL_KW = [
    "allowance", "entitlement", "rate", "per diem", "reimbursement",
    "eligible", "limit", "maximum", "minimum", "₹", "$", "usd", "inr",
    "days", "policy number", "grade", "band", "level", "tier",
    "hotel", "meal", "daily", "monthly", "annual", "laptop", "budget",
    "per night", "per day", "salary", "ctc", "hike", "increment",
]
_COMPUTE_KW = [
    "calculate", "how much", "total", "compute", "sum", "multiply",
    "what is the cost", "how many days", "pro-rata", "cost of",
]

_PLANNER_SYSTEM = """You are a query router for a corporate policy chatbot.
Classify the user query into EXACTLY ONE of:
- "sql"       : requires exact lookup of structured policy data (rates, limits, grades, entitlements)
- "retrieval" : requires reading policy text for procedures, processes, eligibility rules
- "compute"   : arithmetic on already-known values (no lookup needed)
- "direct"    : simple conversational answer, no lookup needed

Also extract:
- "grade": employee grade mentioned (L1-L7, VP, SVP) or null

Respond ONLY with valid JSON: {"route": "...", "grade": "...", "reason": "..."}
No markdown, no extra text."""


def _detect_grade(text: str) -> str | None:
    t = text.lower()
    for key, val in _GRADE_MAP.items():
        if re.search(r"\b" + re.escape(key) + r"\b", t):
            return val
    return None


def _keyword_route(query: str) -> str:
    q = query.lower()
    if any(k in q for k in _SQL_KW):
        return "sql"
    if any(k in q for k in _COMPUTE_KW):
        return "compute"
    return "retrieval"


def _llm_route(query: str, history: list) -> dict:
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=120)
        msgs = [{"role": "system", "content": _PLANNER_SYSTEM}]
        msgs += history[-4:]
        msgs.append({"role": "user", "content": query})
        raw = llm.invoke(msgs).content.strip()
        raw = re.sub(r"```[a-z]*|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning("LLM planner fallback to keywords: %s", e)
        return {"route": _keyword_route(query), "grade": _detect_grade(query), "reason": "keyword"}


def run(state: AgentState) -> AgentState:
    query       = state.get("query", "")
    history     = state.get("history", [])
    retry_count = state.get("retry_count", 0)

    # Augment query with prior failure info on retry
    effective_query = query
    if retry_count > 0 and state.get("verification_issues"):
        issues = "; ".join(state["verification_issues"])
        effective_query = f"{query}\n[RETRY {retry_count}: prior issues: {issues}]"

    decision = _llm_route(effective_query, history)
    route  = decision.get("route", "retrieval")
    grade  = decision.get("grade") or _detect_grade(query) or state.get("employee_grade")

    logger.info("Planner → route=%s grade=%s retry=%d", route, grade, retry_count)

    return trace({
        **state,
        "route": route,
        "employee_grade": grade,
        "needs_compute": route in ("sql", "compute"),
        "retry_count": retry_count,
    }, node="planner", data={"route": route, "grade": grade})
