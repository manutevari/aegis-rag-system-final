"""
Verifier Node — Simplified, safe, no retry explosion
"""

import logging, re
from typing import List, Tuple
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

_NUM  = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_CODE = re.compile(r"\b[A-Z]{1,4}[-_]?\d{2,6}\b")
_HEDGE = ["i believe","i think","probably","approximately","roughly","typically","usually","i'm not sure"]


def _nums(text: str) -> List[float]:
    out = []
    for m in _NUM.finditer(text):
        try:
            out.append(float(m.group().replace(",", "")))
        except:
            pass
    return out


def _check_numbers(answer: str, context: str) -> Tuple[bool, List[str]]:
    ctx_nums = set(_nums(context))
    issues = []
    for n in _nums(answer):
        if n <= 10:
            continue
        if not any(abs(n - c) <= 1 for c in ctx_nums):
            issues.append(f"Number {n:,.0f} not in context")
    return not issues, issues


def _check_codes(answer: str, context: str) -> Tuple[bool, List[str]]:
    ctx_codes = set(_CODE.findall(context))
    issues = [f"Code {c} not in context" for c in _CODE.findall(answer) if c not in ctx_codes]
    return not issues, issues


def _check_compute(answer: str, state: AgentState) -> Tuple[bool, List[str]]:
    result = state.get("compute_result")
    if result is None:
        return True, []
    if not any(abs(n - result) <= 1 for n in _nums(answer)):
        return False, [f"Missing computed value ₹{result:,.2f}"]
    return True, []


def run(state: AgentState) -> AgentState:
    answer  = state.get("answer", "")
    context = state.get("context", "")

    ok1, i1 = _check_numbers(answer, context)
    ok2, i2 = _check_codes(answer, context)
    ok3, i3 = _check_compute(answer, state)

    issues = i1 + i2 + i3
    verified = ok1 and ok2 and ok3

    logger.info("Verifier — verified=%s issues=%s", verified, issues or "none")

    return trace(
        {
            **state,
            "verified": verified,
            "verification_issues": issues,
            "retry": False   # ✅ CRITICAL: stops infinite retry loop
        },
        node="verify",
        data={"verified": verified, "issues": issues}
    )
