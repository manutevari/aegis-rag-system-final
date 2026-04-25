"""Pure-Python deterministic arithmetic. Zero LLM involvement."""
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool
import math, re


@tool
def calculator(expression: str) -> str:
    """MANDATORY: use for ALL numeric calculations"""
    try:
        if not expression:
            return "No expression provided for calculation."

        expression = expression.lower()
        expression = expression.replace("square root of", "sqrt")
        expression = expression.replace("square of", "**2")

        # safer cleanup
        expression = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expression)

        allowed = {
            "sqrt": math.sqrt,
            "pow": pow
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# Core deterministic functions
# ─────────────────────────────────────────────────────────────

def compute_per_diem(days: int, rate: float) -> float:
    return round(days * rate, 2)


def compute_hotel_entitlement(nights: int, rate: float) -> float:
    return round(nights * rate, 2)


def summarise_computation(steps: List[str], total: Optional[float]) -> str:
    return "\n".join(steps) if steps else ""


# ─────────────────────────────────────────────────────────────
# Fallback calculation (calculator + approximation)
# ─────────────────────────────────────────────────────────────

def fallback_calculation(nums: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    """
    Handles calculator fallback with approximation if expression missing
    """
    expr = nums.get("expression")

    # Case 1: exact calculation
    if expr and any(c.isdigit() for c in str(expr)):
        result = calculator(str(expr))
        return None, [f"Calculator used: {expr} = {result}"]

    # Case 2: approximation
    approx_reason = "No explicit expression found. Attempting approximate estimation."
    numbers = re.findall(r"\d+", str(nums))

    if len(numbers) >= 2:
        approx = int(numbers[0]) * int(numbers[1])
        return approx, [
            f"{approx_reason} Approximated using multiplication: {numbers[0]} × {numbers[1]} = {approx}"
        ]

    return None, [f"{approx_reason} Unable to compute due to insufficient data."]


# ─────────────────────────────────────────────────────────────
# Travel allowance computation
# ─────────────────────────────────────────────────────────────

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
            steps.append(f"Meals: ₹{amt}")

        elif cat == "hotel" and row.get("per_night_inr"):
            amt = round(nights * row["per_night_inr"], 2)
            total += amt
            steps.append(f"Hotel: ₹{amt}")

    if total == 0:
        return None, []

    steps.append(f"TOTAL: ₹{total}")
    return total, steps
