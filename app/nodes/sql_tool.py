"""Structured lookup node for the corporate policy database."""

import logging
import re

from app.state import AgentState
from app.tools.sql import PolicyDatabase
from app.utils.tracing import trace

logger = logging.getLogger(__name__)
_db: PolicyDatabase | None = None


def _get_db() -> PolicyDatabase:
    global _db
    if _db is None:
        _db = PolicyDatabase()
    return _db


def _detect_grade(query: str, state_grade: str | None) -> str | None:
    match = re.search(r"\b(L[1-7]|VP|SVP|CXO|Executive)\b", query, flags=re.I)
    if match:
        value = match.group(1).upper()
        return "VP" if value == "EXECUTIVE" else value
    return state_grade


def _detect_travel_type(query: str) -> str | None:
    lower = query.lower()
    if "international" in lower or "overseas" in lower:
        return "international"
    if "domestic" in lower:
        return "domestic"
    if "local" in lower or "cab" in lower or "mileage" in lower:
        return "local"
    return None


def _detect_category(query: str) -> str | None:
    lower = query.lower()
    category_map = {
        "hotel": ["hotel", "lodging", "accommodation", "night"],
        "meal": ["meal", "food", "lunch", "dinner", "breakfast"],
        "transport": ["transport", "cab", "taxi", "mileage", "fuel", "vehicle"],
        "per_diem": ["per diem", "daily allowance", "incidentals"],
        "laptop": ["laptop", "device", "equipment"],
        "allowance": ["allowance", "entitlement", "reimbursement"],
    }
    for category, keywords in category_map.items():
        if any(keyword in lower for keyword in keywords):
            return category
    return None


def _extract_params(query: str, state_grade: str | None, history: list) -> dict:
    policy_code = None
    match = re.search(r"\b[A-Z]{1,4}-[A-Z0-9]{2,}\b", query, flags=re.I)
    if match:
        policy_code = match.group(0).upper()

    department = None
    dept_match = re.search(r"\b(hr|finance|it|sales|engineering|security)\b", query, flags=re.I)
    if dept_match:
        department = dept_match.group(1).lower()

    return {
        "grade": _detect_grade(query, state_grade),
        "travel_type": _detect_travel_type(query),
        "category": _detect_category(query),
        "policy_code": policy_code,
        "department": department,
    }


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    history = state.get("history", [])
    grade = state.get("employee_grade")

    params = _extract_params(query, grade, history)
    logger.info("SQL node params: %s", params)

    try:
        rows = _get_db().query_policy(
            grade=params.get("grade"),
            travel_type=params.get("travel_type"),
            category=params.get("category"),
            policy_code=params.get("policy_code"),
            department=params.get("department"),
        )
    except Exception as exc:
        logger.error("SQL query failed: %s", exc)
        rows = []

    resolved_grade = params.get("grade") or grade
    return trace(
        {**state, "sql_result": rows, "sql_params": params, "employee_grade": resolved_grade},
        node="sql",
        data={"rows": len(rows), "params": params},
    )
