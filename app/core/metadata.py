# =============================================================================
# AEGIS - METADATA MODULE
# =============================================================================
from __future__ import annotations

import json
import os
import re
import textwrap
import uuid
from functools import lru_cache


@lru_cache(maxsize=4)
def _client(api_key: str):
    from openai import OpenAI

    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not resolved_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    return OpenAI(api_key=resolved_key, timeout=10.0, max_retries=0)


_META_SYSTEM = textwrap.dedent(
    """
    You are a metadata extraction assistant.
    Given a corporate policy document, extract exactly these fields as JSON:
      - document_id     : a short code like "TRV-POL-2025-V1" derived from the text
      - policy_category : one of Travel | HR | Finance | Legal | IT | General
      - policy_owner    : department/team name responsible
      - effective_date  : ISO date YYYY-MM-DD; use "2026-01-01" if not found
      - h1_header       : the top-level document title
      - h2_header       : the most prominent sub-section heading
    Respond ONLY with a valid JSON object. No markdown fences.
"""
).strip()

VALID_CATEGORIES = {"Travel", "HR", "Finance", "Legal", "IT", "General"}


def _clean_category(value: str) -> str:
    cat = (value or "General").strip()
    for valid in VALID_CATEGORIES:
        if cat.lower() == valid.lower():
            return valid
    return "General"


def _clean_iso_date(value: str) -> str:
    clean = (value or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", clean):
        return clean
    return "2026-01-01"


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").split())


def _fallback(data: dict, text: str) -> dict:
    regex_data = extract_metadata_regex(text)
    merged = {**regex_data, **{key: value for key, value in data.items() if value}}
    return {
        "document_id": _clean_text(merged.get("document_id")) or f"DOC-{uuid.uuid4().hex[:6].upper()}",
        "policy_category": _clean_category(merged.get("policy_category", "General")),
        "policy_owner": _clean_text(merged.get("policy_owner")) or "Unknown",
        "effective_date": _clean_iso_date(merged.get("effective_date")),
        "h1_header": _clean_text(merged.get("h1_header")),
        "h2_header": _clean_text(merged.get("h2_header")),
    }


def extract_metadata(text: str, api_key: str = "") -> dict:
    snippet = (text or "")[:3000]
    try:
        resp = _client(api_key).chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": _META_SYSTEM},
                {"role": "user", "content": snippet},
            ],
        )
        data = json.loads(resp.choices[0].message.content.strip())
    except Exception:
        data = {}
    return _fallback(data, text or "")


def extract_metadata_regex(text: str) -> dict:
    """Fast regex-only fallback - no LLM call needed."""
    text = text or ""
    doc_id_match = re.search(r"\b([A-Z]{2,6}-[A-Z]{2,5}-\d{4}-V\d+)\b", text)
    date_match = re.search(r"(?:effective|as of|dated?)[\s:]+(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    owner_match = re.search(
        r"(?:owned by|maintained by|contact|policy owner)[\s:*]+([A-Z][A-Za-z\s\-]+?)[\n,.]",
        text,
        re.IGNORECASE,
    )
    h1_match = re.search(r"^# (.+)$", text, re.MULTILINE)
    h2_match = re.search(r"^## (.+)$", text, re.MULTILINE)

    text_lower = text.lower()
    if any(w in text_lower for w in ["travel", "expense", "per diem", "flight", "hotel", "taxi"]):
        category = "Travel"
    elif any(w in text_lower for w in ["leave", "maternity", "hr policy", "employee", "pto", "sabbatical"]):
        category = "HR"
    elif any(w in text_lower for w in ["finance", "budget", "reimbursement", "invoice"]):
        category = "Finance"
    elif any(w in text_lower for w in ["legal", "contract", "compliance", "privacy"]):
        category = "Legal"
    elif any(w in text_lower for w in ["it policy", "information technology", "security", "vpn", "password", "device"]):
        category = "IT"
    else:
        category = "General"

    return {
        "document_id": doc_id_match.group(1) if doc_id_match else f"DOC-{uuid.uuid4().hex[:6].upper()}",
        "policy_category": category,
        "policy_owner": owner_match.group(1).strip() if owner_match else "Unknown",
        "effective_date": date_match.group(1) if date_match else "2026-01-01",
        "h1_header": h1_match.group(1).strip() if h1_match else "",
        "h2_header": h2_match.group(1).strip() if h2_match else "",
    }
