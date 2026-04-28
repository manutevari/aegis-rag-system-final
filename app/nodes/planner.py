"""Deterministic planner node for offline-first routing."""

import logging
import re
from typing import Any, Dict

from app.utils.tracing import trace

logger = logging.getLogger(__name__)

_GRADE_MAP = {
    "l1": "L1",
    "l2": "L2",
    "l3": "L3",
    "l4": "L4",
    "l5": "L5",
    "l6": "L6",
    "l7": "L7",
    "vp": "VP",
    "svp": "SVP",
    "cxo": "CXO",
    "manager": "L5",
    "senior manager": "L6",
    "analyst": "L2",
    "associate": "L3",
    "consultant": "L4",
    "director": "L7",
    "executive director": "VP",
}

_SQL_KW = [
    "allowance",
    "entitlement",
    "rate",
    "per diem",
    "reimbursement",
    "eligible",
    "limit",
    "maximum",
    "minimum",
    "hotel",
    "meal",
    "daily",
    "monthly",
    "annual",
    "budget",
]

_COMPUTE_KW = ["calculate", "total", "compute", "multiply", "cost"]
_NUMPY_KW = ["mean", "average", "std", "array", "sum"]
_PANDAS_KW = ["csv", "dataframe", "table", "groupby"]
_PLOT_KW = ["plot", "chart", "graph"]


def _detect_grade(text: str):
    lower = text.lower()
    for key, value in _GRADE_MAP.items():
        if re.search(rf"\b{key}\b", lower):
            return value
    return None


def _advanced_route(query: str):
    lower = query.lower()
    if any(keyword in lower for keyword in _NUMPY_KW):
        return "numpy_compute"
    if any(keyword in lower for keyword in _PANDAS_KW):
        return "pandas_query"
    if any(keyword in lower for keyword in _PLOT_KW):
        return "plot_chart"
    return None


def _keyword_route(query: str):
    lower = query.lower()
    if any(keyword in lower for keyword in _SQL_KW):
        return "sql"
    if any(keyword in lower for keyword in _COMPUTE_KW):
        return "compute"
    return "retrieval"


def _llm_route(query: str, history: list):
    route = _advanced_route(query) or _keyword_route(query)
    return {
        "route": route,
        "grade": _detect_grade(query),
        "reason": "rules",
    }


def planner_node(state: dict) -> dict:
    try:
        query = state.get("query", "").strip()
        history = state.get("history", []) or []
        retry = state.get("retry_count", 0)

        if not query:
            return {**state, "route": "direct", "error": "Empty query"}

        decision = _llm_route(query, history)
        route = decision["route"]
        grade = decision.get("grade") or _detect_grade(query) or state.get("employee_grade")

        logger.info("[Planner] route=%s grade=%s", route, grade)

        return trace(
            {
                **state,
                "route": route,
                "employee_grade": grade,
                "needs_compute": route in {"sql", "compute", "numpy_compute"},
                "retry_count": retry,
            },
            node="planner",
            data=decision,
        )

    except Exception as exc:
        logger.error("[Planner Crash] %s", exc)
        return {
            **state,
            "route": "retrieval",
            "employee_grade": None,
            "error": str(exc),
        }


run = planner_node
