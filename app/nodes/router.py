from app.core.models import get_llm

VALID_LABELS = {"factual", "lookup", "compute", "unclear"}

def route_intent(query: str) -> dict:
    llm = get_llm(model_override="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a strict classifier.

Return ONLY one word from:
factual, lookup, compute, unclear

Query: {query}
"""

    try:
        res = llm.invoke(prompt)
        out = getattr(res, "content", "").strip().lower()
    except Exception:
        return {"intent": "unclear"}

    # strict validation (fix)
    if out not in VALID_LABELS:
        out = "unclear"

    return {"intent": out}
