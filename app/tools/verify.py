"""Standalone verification utilities."""

import re
from typing import List, Tuple

_NUM = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")


def extract_numbers(text: str) -> List[float]:
    out = []
    for match in _NUM.finditer(text):
        try:
            out.append(float(match.group().replace(",", "")))
        except ValueError:
            pass
    return out


def _close(a: float, b: float, tolerance: float = 1.0) -> bool:
    return abs(a - b) <= tolerance


def _is_derived_number(target: float, answer_numbers: List[float], context_numbers: List[float]) -> bool:
    """Allow simple totals computed from a context value and answer multiplier."""
    for context_number in context_numbers:
        for answer_number in answer_numbers:
            if answer_number == target:
                continue
            if _close(context_number * answer_number, target):
                return True
            if answer_number and _close(context_number / answer_number, target):
                return True
            if _close(context_number + answer_number, target):
                return True
            if _close(context_number - answer_number, target):
                return True
    return False


def verify_numerical_consistency(answer: str, context: str) -> Tuple[bool, List[str]]:
    context_numbers = extract_numbers(context)
    answer_numbers = extract_numbers(answer)
    issues = []

    for number in answer_numbers:
        if number <= 10:
            continue
        if any(_close(number, context_number) for context_number in context_numbers):
            continue
        if _is_derived_number(number, answer_numbers, context_numbers):
            continue
        issues.append(f"{number:,.0f} not in context")

    return not issues, issues


def verify_no_fabrication(answer: str) -> Tuple[bool, List[str]]:
    hedges = ["i believe", "i think", "probably", "approximately", "roughly", "typically", "usually"]
    issues = [f'Hedge: "{phrase}"' for phrase in hedges if phrase in answer.lower()]
    return not issues, issues
