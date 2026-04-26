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
        out = llm.invoke(prompt).content.strip().lower()
    except Exception:
        return {"intent": "unclear"}

    # strict validation
    if out not in VALID_LABELS:
        return {"intent": "unclear"}

    return {"intent": out}
