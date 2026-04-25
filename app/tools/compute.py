"""Pure-Python deterministic arithmetic. Zero LLM involvement."""
from typing import Any, Dict, List, Optional, Tuple

# ✅ ADD THIS BLOCK HERE (exactly here)
from langchain.tools import tool
import math, re

@tool
def calculator(expression: str) -> str:
    """MANDATORY: use for ALL numeric calculations"""
    try:
        expression = expression.lower()
        expression = expression.replace("square root of", "sqrt")
        expression = expression.replace("square of", "**2")
        expression = re.sub(r"[^\d\.\+\-\*\/\(\)\s\w]", "", expression)

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
