"""Standalone verification utilities."""
import re
from typing import List, Tuple

_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")

def extract_numbers(text: str) -> List[float]:
    out = []
    for m in _NUM.finditer(text):
        try: out.append(float(m.group().replace(",","")))
        except: pass
    return out

def verify_numerical_consistency(answer: str, context: str) -> Tuple[bool, List[str]]:
    ctx = set(extract_numbers(context))
    issues = [f"{n:,.0f} not in context" for n in extract_numbers(answer)
              if n > 10 and not any(abs(n-c)<=1 for c in ctx)]
    return not issues, issues

def verify_no_fabrication(answer: str) -> Tuple[bool, List[str]]:
    HEDGES = ["i believe","i think","probably","approximately","roughly","typically","usually"]
    issues = [f'Hedge: "{p}"' for p in HEDGES if p in answer.lower()]
    return not issues, issues
