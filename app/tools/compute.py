"""
Deterministic arithmetic helpers for policy computations.

This module is intentionally free of LLM calls so the compute node and unit
suite can use it without network access.
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression."""
    try:
        if not expression:
            return "No expression provided."

        normalized = expression.lower()
        normalized = normalized.replace("square root of", "sqrt")
        normalized = normalized.replace("square of", "**2")
        normalized = re.sub(r"[^0-9\.\+\-\*\/\(\)\s]", "", normalized)

        result = eval(
            normalized,
            {"__builtins__": {}},
            {"sqrt": math.sqrt, "pow": pow},
        )
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


def compute_per_diem(days: int, rate: float) -> float:
    return round(days * rate, 2)


def compute_hotel_entitlement(nights: int, rate: float) -> float:
    return round(nights * rate, 2)


def compute_reimbursement(days: int, per_diem_rate: float, nights: int, hotel_rate: float) -> float:
    return round(compute_per_diem(days, per_diem_rate) + compute_hotel_entitlement(nights, hotel_rate), 2)


def compute_leave_encashment(monthly_salary: float, leave_days: int) -> float:
    return round((monthly_salary / 26) * leave_days, 2)


def compute_variable_pay(annual_pay: float, percentage: float) -> float:
    return round(annual_pay * (percentage / 100), 2)


def compute_pro_rata(amount: float, active_days: int, total_days: int) -> float:
    if total_days <= 0:
        return 0.0
    return round(amount * (active_days / total_days), 2)


def compute_hra(basic_salary: float, city_type: str) -> float:
    rate = 0.50 if city_type.lower() == "metro" else 0.40
    return round(basic_salary * rate, 2)


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
