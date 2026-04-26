"""
Hybrid Intent Router

- Fast rule-based classification (cheap + instant)
- LLM fallback for ambiguous queries
- Strict label enforcement
"""

from app.core.models import get_llm

VALID_LABELS = {"chat", "rag", "compute", "unclear"}


# ─────────────────────────────────────────────────────────────
# 🔹 Rule-Based Fast Path
# ─────────────────────────────────────────────────────────────

def _rule_based_intent(query: str) -> str:
    q = query.lower()

    if any(x in q for x in ["hi", "hello", "how are you", "who are you"]):
        return "chat"

    if any(x in q for x in ["calculate", "total", "cost", "₹", "percent"]):
        return "compute"

    if any(x in q for x in ["policy", "allowance", "rule", "eligibility"]):
        return "rag"

    return "unclear"


# ─────────────────────────────────────────────────────────────
# 🔹 LLM Fallback (Strict)
# ─────────────────────────────────────────────────────────────

def _llm_intent(query: str) -> str:
    llm = get_llm(model_override="gpt-4o-mini", temperature=0)

    prompt = f"""
You are a strict classifier.

Return ONLY one word from:
chat, rag, compute, unclear

Query: {query}
"""

    try:
        res = llm.invoke(prompt)
        out = getattr(res, "content", "").strip().lower()
    except Exception:
        return "unclear"

    if out not in VALID_LABELS:
        return "unclear"

    return out


# ─────────────────────────────────────────────────────────────
# 🔹 Final Hybrid Classifier
# ─────────────────────────────────────────────────────────────

def classify_intent(query: str) -> str:
    # 1️⃣ fast path
    rule_intent = _rule_based_intent(query)

    # if confident → return immediately
    if rule_intent != "unclear":
        return rule_intent

    # 2️⃣ fallback to LLM
    return _llm_intent(query)


# ─────────────────────────────────────────────────────────────
# 🔹 LangGraph Node Wrapper
# ─────────────────────────────────────────────────────────────

def run(state):
    query = state.get("query", "")
    intent = classify_intent(query)

    return {**state, "intent": intent}
