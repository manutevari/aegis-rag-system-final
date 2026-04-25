"""
SQL Tool Node — Structured lookup against the corporate policy database.

Extracts query parameters (grade, category, travel_type, etc.) via LLM,
then executes a parameterised SQL query. Grade is the primary filter key
because almost every policy table in corporate HR is indexed by grade/band.
"""

import json
import logging
import os
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


_EXTRACT_SYSTEM = """Extract policy lookup parameters from the user query as JSON.
Fields (use null if not present):
- grade: employee grade string, e.g. "L4", "VP", "L6-L7"
- travel_type: "domestic" | "international" | "local" | null
- category: "hotel" | "meal" | "transport" | "per_diem" | "laptop" | "allowance" | null
- policy_code: explicit policy code like "T-04" | null
- department: department name | null

Return ONLY a JSON object. No markdown."""


def _extract_params(query: str, state_grade: str | None, history: list) -> dict:
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=150)
        msgs = [{"role": "system", "content": _EXTRACT_SYSTEM}]
        msgs += history[-2:]
        msgs.append({"role": "user", "content": query})
        raw = llm.invoke(msgs).content.strip()
        raw = re.sub(r"```[a-z]*|```", "", raw).strip()
        params = json.loads(raw)
        # Prefer state-detected grade over LLM extraction
        if state_grade and not params.get("grade"):
            params["grade"] = state_grade
        return params
    except Exception as e:
        logger.warning("SQL param extraction failed: %s", e)
        return {"grade": state_grade}


def run(state: AgentState) -> AgentState:
    query   = state.get("query", "")
    history = state.get("history", [])
    grade   = state.get("employee_grade")

    params = _extract_params(query, grade, history)
    logger.info("SQL node — params: %s", params)

    try:
        rows = _get_db().query_policy(
            grade=params.get("grade"),
            travel_type=params.get("travel_type"),
            category=params.get("category"),
            policy_code=params.get("policy_code"),
            department=params.get("department"),
        )
    except Exception as e:
        logger.error("SQL query failed: %s", e)
        rows = []

    # Persist resolved grade back to state
    resolved_grade = params.get("grade") or grade
    return trace(
        {**state, "sql_result": rows, "sql_params": params, "employee_grade": resolved_grade},
        node="sql", data={"rows": len(rows), "params": params},
    )
