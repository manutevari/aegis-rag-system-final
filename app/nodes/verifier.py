"""
Verifier Node — Blocking quality gate before HITL.

Four checks:
  1. Numerical grounding   — every number in answer exists in context
  2. Fabrication phrases   — flags hedging language ("I think", "approximately")
  3. Policy code validity  — every cited code exists in context
  4. Compute consistency   — if FINAL COMPUTED VALUE exists, answer must use it
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
        try: out.append(float(m.group().replace(",", "")))
        except: pass
    return out


def _check_numbers(answer: str, context: str) -> Tuple[bool, List[str]]:
    ctx_nums = set(_nums(context))
    issues = []
    for n in _nums(answer):
        if n <= 10: continue
        if not any(abs(n - c) <= 1 for c in ctx_nums):
            issues.append(f"Number {n:,.0f} not found in context")
    return not issues, issues


def _check_fabrication(answer: str) -> Tuple[bool, List[str]]:
    low = answer.lower()
    issues = [f'Hedge phrase: "{p}"' for p in _HEDGE if p in low]
    return not issues, issues


def _check_codes(answer: str, context: str) -> Tuple[bool, List[str]]:
    ctx_codes = set(_CODE.findall(context))
    issues = [f"Code {c} not in context" for c in _CODE.findall(answer) if c not in ctx_codes]
    return not issues, issues


def _check_compute(answer: str, state: AgentState) -> Tuple[bool, List[str]]:
    result = state.get("compute_result")
    if result is None: return True, []
    if not any(abs(n - result) <= 1 for n in _nums(answer)):
        return False, [f"Answer doesn't reflect computed ₹{result:,.2f}"]
    return True, []


def run(state: AgentState) -> AgentState:
    answer  = state.get("answer", "")
    context = state.get("context", "")
    issues: List[str] = []

    ok1, i1 = _check_numbers(answer, context)
    ok2, i2 = _check_fabrication(answer)     # soft warning only
    ok3, i3 = _check_codes(answer, context)
    ok4, i4 = _check_compute(answer, state)
    issues   = i1 + i2 + i3 + i4
    verified = ok1 and ok3 and ok4           # fabrication is advisory

    retry = state.get("retry_count", 0) + (0 if verified else 1)
    logger.info("Verifier — verified=%s issues=%s", verified, issues or "none")

    return trace({**state, "verified": verified, "verification_issues": issues, "retry_count": retry},
                 node="verify", data={"verified": verified, "issues": issues})
