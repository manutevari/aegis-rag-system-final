"""
Generator Node — Strict, grounded LLM answer generation.
"""

import logging, re, time
from app.state import AgentState
from app.utils.tracing import trace
from app.core.models import invoke_llm

logger = logging.getLogger(__name__)

_SYSTEM = """You are a corporate policy assistant.

Answer ONLY using the provided policy context.

Rules:
- Do not invent any values, policies, or assumptions.
- Use exact numbers from context (₹ / USD).
- If a computed value is present, use it exactly.
- Cite policy codes when available.
- If answer is missing, say:
  "This is not covered in the available policy data."

Format:
- Use bullet points for clarity.
- End with: Source: [policy name/code] if available.
"""


# ✅ Minimal safe wrapper (no over-retry)
def safe_invoke_llm(messages, model_override=None):
    try:
        return invoke_llm(messages, model_override=model_override)
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            logger.warning("[LLM] Rate limited")
        else:
            logger.error(f"[LLM] Error: {err}")
        raise


def run(state: AgentState) -> AgentState:
    query   = state.get("query", "")
    context = state.get("context", "")
    history = state.get("history") or []
    grade   = state.get("employee_grade", "")
    model_override = state.get("model")

    grade_note = f" (Employee grade: {grade})" if grade else ""

    try:
        msgs = [{"role": "system", "content": _SYSTEM}]
        msgs += history[-4:]
        msgs.append({
            "role": "user",
            "content": f"POLICY CONTEXT:\n{context}\n\nQUESTION: {query}{grade_note}\n\nAnswer using ONLY the context above."
        })

        response = safe_invoke_llm(msgs, model_override=model_override)

        # ✅ Robust extraction
        answer = getattr(response, "content", None)
        if not answer:
            try:
                answer = response.choices[0].message.content
            except Exception:
                answer = str(response)

        answer = answer.strip()

    except Exception as e:
        logger.error("Generation error: %s", e)
        answer = "⚠️ System busy or rate-limited. Please try again."

    # ✅ Extract sources
    sources = re.findall(r"Source:\s*(.+)", answer)

    return trace(
        {
            **state,
            "answer": answer,
            "sources": [s.strip() for s in sources],
            "retry": False  # ✅ HARD STOP (no loop)
        },
        node="generate",
        data={
            "len": len(answer),
            "model": model_override or "default"
        }
    )
