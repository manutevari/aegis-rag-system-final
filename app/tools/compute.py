"""Pure-Python deterministic arithmetic. Zero LLM involvement."""
from typing import Any, Dict, List, Optional, Tuple

def compute_per_diem(days: int, rate: float) -> float:
    return round(days * rate, 2)

def compute_hotel_entitlement(nights: int, rate: float) -> float:
    return round(nights * rate, 2)

def compute_reimbursement(days, meal_rate, nights, hotel_rate, transport=0.0, misc=0.0) -> float:
    return round(days*meal_rate + nights*hotel_rate + transport + misc, 2)

def compute_leave_encashment(basic_monthly: float, days: int) -> float:
    return round((basic_monthly / 26) * days, 2)

def compute_variable_pay(annual_basic: float, pct: float) -> float:
    return round(annual_basic * pct / 100, 2)

def compute_pro_rata(amount: float, worked: int, total: int) -> float:
    return round(amount * worked / total, 2) if total else 0.0

def compute_hra(basic_monthly: float, city: str = "metro") -> float:
    return round(basic_monthly * (0.50 if city.lower()=="metro" else 0.40), 2)

def compute_travel_allowance(rows: List[dict], nums: Dict[str, Any]) -> Tuple[Optional[float], List[str]]:
    days   = nums.get("days", 0)
    nights = nums.get("nights", days)
    steps, total = [], 0.0
    computed = False
    for row in rows:
        cat = (row.get("category") or "").lower()
        if cat in ("meal","per_diem") and days and row.get("per_day_inr"):
            amt = compute_per_diem(days, row["per_day_inr"])
            total += amt
            steps.append(f"Meals: {days}d × ₹{row['per_day_inr']:,.0f} = ₹{amt:,.2f} (Policy {row.get('policy_code','')})")
            computed = True
        elif cat == "hotel" and nights and row.get("per_night_inr"):
            amt = compute_hotel_entitlement(nights, row["per_night_inr"])
            total += amt
            steps.append(f"Hotel: {nights}n × ₹{row['per_night_inr']:,.0f} = ₹{amt:,.2f} (Policy {row.get('policy_code','')})")
            computed = True
        elif cat == "transport" and days and row.get("per_day_inr"):
            amt = round(days * row["per_day_inr"], 2)
            total += amt
            steps.append(f"Transport: {days}d × ₹{row['per_day_inr']:,.0f} = ₹{amt:,.2f} (Policy {row.get('policy_code','')})")
            computed = True
    if not computed:
        return None, []
    steps += ["─"*40, f"TOTAL: ₹{total:,.2f}"]
    return round(total, 2), steps

def summarise_computation(steps: List[str], total: Optional[float]) -> str:
    return "\n".join(steps) if steps else ""
