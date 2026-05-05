"""
Deterministic arithmetic helpers for policy computations.

This module is intentionally free of LLM calls so the compute node and unit
suite can use it without network access.
"""

import ast
import re
from decimal import Decimal, DivisionByZero, InvalidOperation, getcontext
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool

getcontext().prec = 28


def _decimal(value: Any) -> Decimal:
    return Decimal(str(value).replace(",", ""))


def _normalise_expression(expression: str) -> str:
    normalized = (expression or "").lower()
    normalized = normalized.replace("×", "*").replace("÷", "/")
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d[\d,]*(?:\.\d+)?)", r"(\1/100)*\2", normalized)
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"(\1/100)", normalized)
    return re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", normalized).strip()


def _safe_decimal_eval(expression: str) -> Decimal:
    operators = {
        ast.Add: lambda left, right: left + right,
        ast.Sub: lambda left, right: left - right,
        ast.Mult: lambda left, right: left * right,
        ast.Div: lambda left, right: left / right,
        ast.Pow: lambda left, right: left ** int(right),
    }

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -evaluate(node.operand)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd):
            return evaluate(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError("Unsupported expression.")

    return evaluate(ast.parse(expression, mode="eval"))


def _format_decimal(value: Decimal) -> str:
    text = format(value.normalize(), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression with Decimal precision."""
    try:
        if not expression:
            return "No expression provided."

        normalized = _normalise_expression(expression)
        if not normalized:
            return "No valid expression found."
        return _format_decimal(_safe_decimal_eval(normalized))
    except (SyntaxError, ValueError, InvalidOperation, DivisionByZero, OverflowError) as e:
        return f"Calculation error: {str(e)}"


def compute_per_diem(days: int, rate: float) -> float:
    return float((_decimal(days) * _decimal(rate)).quantize(Decimal("0.01")))


def compute_hotel_entitlement(nights: int, rate: float) -> float:
    return float((_decimal(nights) * _decimal(rate)).quantize(Decimal("0.01")))


def compute_reimbursement(days: int, per_diem_rate: float, nights: int, hotel_rate: float) -> float:
    return round(compute_per_diem(days, per_diem_rate) + compute_hotel_entitlement(nights, hotel_rate), 2)


def compute_leave_encashment(monthly_salary: float, leave_days: int) -> float:
    return float(((_decimal(monthly_salary) / Decimal("26")) * _decimal(leave_days)).quantize(Decimal("0.01")))


def compute_variable_pay(annual_pay: float, percentage: float) -> float:
    return float((_decimal(annual_pay) * (_decimal(percentage) / Decimal("100"))).quantize(Decimal("0.01")))


def compute_pro_rata(amount: float, active_days: int, total_days: int) -> float:
    if total_days <= 0:
        return 0.0
    return float((_decimal(amount) * (_decimal(active_days) / _decimal(total_days))).quantize(Decimal("0.01")))


def compute_hra(basic_salary: float, city_type: str) -> float:
    rate = Decimal("0.50") if city_type.lower() == "metro" else Decimal("0.40")
    return float((_decimal(basic_salary) * rate).quantize(Decimal("0.01")))


def compute_travel_allowance(rows, nums):
    total = 0.0
    steps = []

    days = nums.get("days", 0)
    nights = nums.get("nights", days)

    for row in rows:
        cat = (row.get("category") or "").lower()

        if cat in ("meal", "per_diem") and row.get("per_day_inr"):
            amount = round(days * row["per_day_inr"], 2)
            total += amount
            steps.append(f"Meals: INR {amount:,.2f}")

        elif cat == "hotel" and row.get("per_night_inr"):
            amount = round(nights * row["per_night_inr"], 2)
            total += amount
            steps.append(f"Hotel: INR {amount:,.2f}")

    if total == 0:
        return None, []

    steps.append(f"TOTAL: INR {total:,.2f}")
    return total, steps


def summarise_computation(steps: List[str]) -> str:
    return "\n".join(steps) if steps else "No computation performed."


def extract_nums(query: str) -> Dict[str, Any]:
    q = query.lower()
    result: Dict[str, Any] = {}

    percent_of = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of\s+(\d[\d,]*(?:\.\d+)?)", q)
    if percent_of:
        result["expression"] = f"({percent_of.group(1)}/100)*{percent_of.group(2).replace(',', '')}"

    arithmetic = re.findall(r"[\d,\s\.\+\-\*\/\(\)%]{3,}", q)
    arithmetic = [item.strip() for item in arithmetic if any(op in item for op in ["+", "-", "*", "/", "%"])]
    if arithmetic and "expression" not in result:
        result["expression"] = max(arithmetic, key=len)

    m = re.search(r"(\d+)\s*(?:night|nights)", q)
    if m:
        result["nights"] = int(m.group(1))

    m = re.search(r"(\d+)\s*(?:day|days)", q)
    if m:
        result["days"] = int(m.group(1))

    m = re.search(r"[rs\$\.]*\s*(\d[\d,]+)", q)
    if m:
        result["rate"] = float(m.group(1).replace(",", ""))

    m = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
    if m:
        result["percentage"] = float(m.group(1))

    return result


def fallback_calculation(nums: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    expr = nums.get("expression")

    if expr and any(c.isdigit() for c in str(expr)):
        result = calculator.invoke({"expression": str(expr)})
        if not str(result).startswith("Calculation error"):
            return float(result), [f"Calculator: {_normalise_expression(str(expr))} = {result}"]
        return None, [f"Calculator: {expr} = {result}"]

    numbers = re.findall(r"\d+", str(nums))

    if len(numbers) >= 2:
        approx = int(numbers[0]) * int(numbers[1])
        return approx, [f"Approximation: {numbers[0]} x {numbers[1]} = {approx}"]

    return None, ["Insufficient data for computation."]


def compute(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state.get("query", "")
    sql_rows = state.get("sql_result") or []

    nums = extract_nums(query)

    steps: List[str] = []
    final: Optional[float] = None

    if sql_rows:
        final, steps = compute_travel_allowance(sql_rows, nums)

    if final is None and nums:
        days = nums.get("days", 0)
        nights = nums.get("nights", days)
        rate = nums.get("rate", 0.0)

        if days and rate:
            final = compute_per_diem(days, rate)
            steps.append(f"Per diem: {days} x INR {rate:,.2f} = INR {final:,.2f}")

        elif nights and rate:
            final = compute_hotel_entitlement(nights, rate)
            steps.append(f"Hotel: {nights} x INR {rate:,.2f} = INR {final:,.2f}")

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
