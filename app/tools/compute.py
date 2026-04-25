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

        # safer cleanup (removed \w to avoid unsafe names)
        expression = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", expression)

        allowed = {
            "sqrt": math.sqrt,
            "pow": pow
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ⬇️ KEEP ALL YOUR EXISTING CODE EXACTLY SAME BELOW

def compute_per_diem(days: int, rate: float) -> float:
    return round(days * rate, 2)


# ✅ FIXED FALLBACK (connect + approx handling)

def fallback_calculation(nums: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    """
    Handles calculator fallback with approximation if expression missing
    """
    expr = nums.get("expression")

    # 🔹 Case 1: Proper expression exists
    if expr and any(c.isdigit() for c in str(expr)):
        result = calculator(str(expr))
        return None, [f"Calculator used: {expr} = {result}"]

    # 🔹 Case 2: No expression → approximate reasoning
    approx_reason = "No explicit expression found. Attempting approximate estimation."

    # try extracting numbers
    numbers = re.findall(r"\d+", str(nums))
    if len(numbers) >= 2:
        approx = int(numbers[0]) * int(numbers[1])
        return approx, [f"{approx_reason} Approximated using multiplication: {numbers[0]} × {numbers[1]} = {approx}"]

    return None, [f"{approx_reason} Unable to compute due to insufficient data."]
