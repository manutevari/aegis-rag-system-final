# ✅ Export alias for workflow compatibility
run = planner_node
"""
Planner Node — Production Hardened

Fixes:
- Callable contract (planner_node)
- Safe routing (no crashes on LLM failure)
- Deterministic priority routing
- Strict JSON parsing
- Type-safe output

Routing:
  sql | retrieval | compute | direct
  numpy_compute | pandas_query | plot_chart
"""

import json
import logging
import os
import re
from typing import Dict, Any

from app.utils.tracing import trace

logger = logging.getLogger(__name__)

# ==============================
# 🔹 CONSTANTS
# ==============================

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
    "hotel", "meal", "daily", "monthly", "annual", "budget",
]

_COMPUTE_KW = [
    "calculate", "total", "compute", "multiply", "cost",
]

_NUMPY_KW = ["mean", "average", "std", "array", "sum"]
_PANDAS_KW = ["csv", "dataframe", "table", "groupby"]
_PLOT_KW = ["plot", "chart", "graph"]

_ALLOWED_ROUTES = {
    "sql", "retrieval", "compute", "direct",
    "numpy_compute", "pandas_query", "plot_chart"
}

_PLANNER_SYSTEM = """Classify the query into ONE:
sql | retrieval | compute | direct

Also extract:
grade: L1-L7, VP, SVP or null

Return ONLY valid JSON:
{"route": "...", "grade": "..."}
"""

# ==============================
# 🔹 HELPERS
# ==============================

def _safe_json_load(raw: str) -> Dict[str, Any]:
    try:
        raw = re.sub(r"```[a-z]*|```", "", raw).strip()
        return json.loads(raw)
    except Exception:
        return {}

def _detect_grade(text: str):
    t = text.lower()
    for k, v in _GRADE_MAP.items():
        if re.search(rf"\b{k}\b", t):
            return v
    return None

def _advanced_route(q: str):
    q = q.lower()

    if any(k in q for k in _NUMPY_KW):
        return "numpy_compute"

    if any(k in q for k in _PANDAS_KW):
        return "pandas_query"

    if any(k in q for k in _PLOT_KW):
        return "plot_chart"

    return None

def _keyword_route(q: str):
    q = q.lower()

    if any(k in q for k in _SQL_KW):
        return "sql"

    if any(k in q for k in _COMPUTE_KW):
        return "compute"

    return "retrieval"

def _llm_route(query: str, history: list):
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            max_tokens=80
        )

        msgs = [{"role": "system", "content": _PLANNER_SYSTEM}]
        msgs += history[-3:]
        msgs.append({"role": "user", "content": query})

        raw = llm.invoke(msgs).content
        parsed = _safe_json_load(raw)

        route = parsed.get("route")
        grade = parsed.get("grade")

        if route not in _ALLOWED_ROUTES:
            raise ValueError("Invalid route")

        return {
            "route": route,
            "grade": grade,
            "reason": "llm"
        }

    except Exception as e:
        logger.warning(f"Planner fallback → {e}")

        return {
            "route": _keyword_route(query),
            "grade": _detect_grade(query),
            "reason": "fallback"
        }

# ==============================
# 🔹 MAIN NODE (FINAL)
# ==============================

def planner_node(state: dict) -> dict:
    try:
        query = state.get("query", "").strip()
        history = state.get("history", []) or []
        retry = state.get("retry_count", 0)

        if not query:
            return {
                **state,
                "route": "direct",
                "error": "Empty query"
            }

        # STEP 1 — Advanced routing (highest priority)
        route = _advanced_route(query)

        if route:
            decision = {
                "route": route,
                "grade": None,
                "reason": "advanced"
            }
        else:
            # STEP 2 — LLM / fallback
            decision = _llm_route(query, history)
            route = decision["route"]

        # STEP 3 — Grade resolution
        grade = (
            decision.get("grade")
            or _detect_grade(query)
            or state.get("employee_grade")
        )

        logger.info(f"[Planner] route={route} grade={grade}")

        return trace({
            **state,
            "route": route,
            "employee_grade": grade,
            "needs_compute": route in {
                "sql", "compute", "numpy_compute"
            },
            "retry_count": retry
        }, node="planner", data=decision)

    except Exception as e:
        logger.error(f"[Planner Crash] {e}")

        # 🔥 HARD FAILSAFE (never break graph)
        return {
            **state,
            "route": "retrieval",
            "employee_grade": None,
            "error": str(e)
        }
