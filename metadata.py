# =============================================================================
# AEGIS — METADATA MODULE
# =============================================================================
from __future__ import annotations
import json, re, textwrap, uuid, os
from functools import lru_cache

@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

_META_SYSTEM = textwrap.dedent("""
    You are a metadata extraction assistant.
    Given a corporate policy document, extract exactly these fields as JSON:
      - document_id     : a short code like "TRV-POL-2025-V1" derived from the text
      - policy_category : one of Travel | HR | Finance | Legal | IT | General
      - policy_owner    : department/team name responsible
      - effective_date  : ISO date YYYY-MM-DD; use "2026-01-01" if not found
      - h1_header       : the top-level document title
      - h2_header       : the most prominent sub-section heading
    Respond ONLY with a valid JSON object. No markdown fences.
""").strip()

VALID_CATEGORIES = {"Travel", "HR", "Finance", "Legal", "IT", "General"}

def extract_metadata(text: str) -> dict:
    snippet = text[:3000]
    try:
        resp = _client().chat.completions.create(
            model="gpt-4o-mini", temperature=0,
            messages=[
                {"role": "system", "content": _META_SYSTEM},
                {"role": "user",   "content": snippet},
            ],
        )
        data = json.loads(resp.choices[0].message.content.strip())
    except Exception:
        data = {}

    cat = data.get("policy_category", "General")
    if cat not in VALID_CATEGORIES:
        cat = "General"

    return {
        "document_id":     data.get("document_id",     f"DOC-{uuid.uuid4().hex[:6].upper()}"),
        "policy_category": cat,
        "policy_owner":    data.get("policy_owner",    "Unknown"),
        "effective_date":  data.get("effective_date",  "2026-01-01"),
        "h1_header":       data.get("h1_header",       ""),
        "h2_header":       data.get("h2_header",       ""),
    }


def extract_metadata_regex(text: str) -> dict:
    """Fast regex-only fallback — no LLM call needed."""
    doc_id_match = re.search(r"\b([A-Z]{2,6}-[A-Z]{2,5}-\d{4}-V\d+)\b", text)
    date_match   = re.search(r"(?:effective|as of|dated?)[\s:]+(\d{4}-\d{2}-\d{2})", text, re.IGNORECASE)
    owner_match  = re.search(r"(?:owned by|maintained by|contact)[\s:]+([A-Z][A-Za-z\s\-]+?)[\n,.]", text, re.IGNORECASE)
    h1_match     = re.search(r"^# (.+)$", text, re.MULTILINE)
    h2_match     = re.search(r"^## (.+)$", text, re.MULTILINE)

    text_lower = text.lower()
    if any(w in text_lower for w in ["travel", "expense", "per diem", "flight"]):
        category = "Travel"
    elif any(w in text_lower for w in ["leave", "maternity", "hr policy", "employee"]):
        category = "HR"
    elif any(w in text_lower for w in ["finance", "budget", "reimbursement"]):
        category = "Finance"
    else:
        category = "General"

    return {
        "document_id":     doc_id_match.group(1) if doc_id_match else f"DOC-{uuid.uuid4().hex[:6].upper()}",
        "policy_category": category,
        "policy_owner":    owner_match.group(1).strip() if owner_match else "Unknown",
        "effective_date":  date_match.group(1) if date_match else "2026-01-01",
        "h1_header":       h1_match.group(1) if h1_match else "",
        "h2_header":       h2_match.group(1) if h2_match else "",
    }
