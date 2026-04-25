"""
Compute Node — 100% deterministic arithmetic. Zero LLM involvement.

Dispatches to pure-Python functions in app/tools/compute.py.
Extracts numeric parameters from the query using regex, then combines
with SQL rows to produce a fully auditable computation trail.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.state import AgentState
from app.tools.compute import (
    compute_travel_allowance, compute_per_diem,
    compute_hotel_entitlement, summarise_computation,
)
from app.utils.tracing import trace

logger = logging.getLogger(__name__)


def _extract_nums(query: str) -> Dict[str, Any]:
    q = query.lower()
    result: Dict[str, Any] = {}

    m = re.search(r"(\d+)\s*(?:night|nights)", q)
    if m: result["nights"] = int(m.group(1))

    m = re.search(r"(\d+)\s*(?:day|days)", q)
    if m: result["days"] = int(m.group(1))

    m = re.search(r"[₹$rs\.]*\s*(\d[\d,]+)\s*(?:per|/|-)?(?:day|night)?", q)
    if m: result["rate"] = float(m.group(1).replace(",", ""))

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if m: result["percentage"] = float(m.group(1))

    return result


def run(state: AgentState) -> AgentState:
    query    = state.get("query", "")
    sql_rows = state.get("sql_result") or []
    nums     = _extract_nums(query)

    steps: List[str] = []
    final: Optional[float] = None

    if sql_rows:
        final, steps = compute_travel_allowance(sql_rows, nums)

    # Fallback: direct query-number calculation
    if final is None and nums:
        days   = nums.get("days", 0)
        nights = nums.get("nights", days)
        rate   = nums.get("rate", 0.0)
        if days and rate:
            final = compute_per_diem(days, rate)
            steps.append(f"Per diem: {days} days × ₹{rate:,.2f} = ₹{final:,.2f}")
        elif nights and rate:
            final = compute_hotel_entitlement(nights, rate)
            steps.append(f"Hotel: {nights} nights × ₹{rate:,.2f} = ₹{final:,.2f}")

    summary = summarise_computation(steps, final)
    logger.info("Compute — result=%s steps=%d", final, len(steps))

    return trace(
        {**state, "compute_result": final, "compute_steps": steps, "compute_summary": summary},
        node="compute", data={"result": final, "steps": len(steps)},
    )
