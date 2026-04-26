"""
Compute Tool — Deterministic arithmetic engine (NO LLM).

Handles:
- Travel allowance (SQL-driven)
- Per diem / hotel fallback
- Calculator fallback
- Full audit trail (steps + summary)
"""

import re
import math
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool


# ─────────────────────────────────────────────────────────────
# 🔹 SAFE CALCULATOR TOOL
# ─────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Safe arithmetic evaluator"""
    try:
        if not expression:
            return "No expression provided."

        expression = expression.lower()
        expression = expression.replace("square root of", "sqrt")
        expression = expression.replace("square of", "**2")

        expression = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expression)

        allowed = {"sqrt": math.sqrt, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)

        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# 🔹 CORE COMPUTATION FUNCTIONS
# ─────────────────────────────────────────────────────────────

def compute_per_diem(days: int, rate: float) -> float:
    return round(days * rate, 2)


def compute_hotel_entitlement(nights: int, rate: float) -> float:
    return round(nights * rate, 2)


def compute_travel_allowance(rows, nums):
    total = 0.0
    steps = []

    days = nums.get("days", 0)
    nights = nums.get("nights", days)

    for row in rows:
        cat = (row.get("category") or "").lower()

        if cat in ("meal", "per_diem") and row.get("per_day_inr"):
            amt = round(days * row["per_day_inr"], 2)
            total += amt
            steps.append(f"Meals: ₹{amt:,.2f}")

        elif cat == "hotel" and row.get("per_night_inr"):
            amt = round(nights * row["per_night_inr"], 2)
            total += amt
            steps.append(f"Hotel: ₹{amt:,.2f}")

    if total == 0:
        return None, []

    steps.append(f"TOTAL: ₹{total:,.2f}")
    return total, steps


def summarise_computation(steps: List[str]) -> str:
    return "\n".join(steps) if steps else "No computation performed."


# ─────────────────────────────────────────────────────────────
# 🔹 NUMBER EXTRACTION
# ─────────────────────────────────────────────────────────────

def extract_nums(query: str) -> Dict[str, Any]:
    q = query.lower()
    result: Dict[str, Any] = {}

    m = re.search(r"(\d+)\s*(?:night|nights)", q)
    if m: result["nights"] = int(m.group(1))

    m = re.search(r"(\d+)\s*(?:day|days)", q)
    if m: result["days"] = int(m.group(1))

    m = re.search(r"[₹$rs\.]*\s*(\d[\d,]+)", q)
    if m: result["rate"] = float(m.group(1).replace(",", ""))

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if m: result["percentage"] = float(m.group(1))

    return result


# ─────────────────────────────────────────────────────────────
# 🔹 FALLBACK LOGIC
# ─────────────────────────────────────────────────────────────

def fallback_calculation(nums: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    expr = nums.get("expression")

    if expr and any(c.isdigit() for c in str(expr)):
        result = calculator(str(expr))
        return None, [f"Calculator: {expr} = {result}"]

    numbers = re.findall(r"\d+", str(nums))

    if len(numbers) >= 2:
        approx = int(numbers[0]) * int(numbers[1])
        return approx, [f"Approximation: {numbers[0]} × {numbers[1]} = {approx}"]

    return None, ["Insufficient data for computation."]


# ─────────────────────────────────────────────────────────────
# 🔹 MAIN ENTRY (CALLED BY NODE)
# ─────────────────────────────────────────────────────────────

def compute(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    sql_rows = state.get("sql_result") or []

    nums = extract_nums(query)

    steps: List[str] = []
    final: Optional[float] = None

    # 1️⃣ SQL-driven computation
    if sql_rows:
        final, steps = compute_travel_allowance(sql_rows, nums)

    # 2️⃣ Direct computation fallback
    if final is None and nums:
        days = nums.get("days", 0)
        nights = nums.get("nights", days)
        rate = nums.get("rate", 0.0)

        if days and rate:
            final = compute_per_diem(days, rate)
            steps.append(f"Per diem: {days} × ₹{rate:,.2f} = ₹{final:,.2f}")

        elif nights and rate:
            final = compute_hotel_entitlement(nights, rate)
            steps.append(f"Hotel: {nights} × ₹{rate:,.2f} = ₹{final:,.2f}")

    # 3️⃣ Final fallback
    if final is None:
        final, fallback_steps = fallback_calculation(nums)
        steps.extend(fallback_steps)

    summary = summarise_computation(steps)

    return {
        **state,
        "compute_result": final,
        "compute_steps": steps,
        "compute_summary": summary,
    }
