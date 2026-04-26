"""
Generator Node — Strict, grounded LLM answer generation.
"""

import logging, os, re
from app.state import AgentState
from app.utils.tracing import trace
from app.core.models import invoke_llm

logger = logging.getLogger(__name__)

_SYSTEM = """You are an authoritative corporate policy assistant.
Answer using ONLY the policy context provided below. Never invent numbers, rates or policies.

STRICT RULES:
1. Every monetary value (₹ / USD / amount) must come verbatim from the context.
2. If a FINAL COMPUTED VALUE exists in context, use that exact figure.
3. Cite the policy code when available — e.g. "(per Policy T-04)".
4. If information is missing, say exactly: "This is not covered in the available policy data."
5. Use bullet points for multi-part answers. Be concise.
6. End with: Source: [policy name/code] — if available."""


def run(state: AgentState) -> AgentState:
    query   = state.get("query", "")
    context = state.get("context", "")
    history = state.get("history") or []
    grade   = state.get("employee_grade", "")
    model_override = state.get("model")  # from retry controller

    grade_note = f" (Employee grade: {grade})" if grade else ""

    try:
        msgs = [{"role": "system", "content": _SYSTEM}]
        msgs += history[-4:]
        msgs.append({
            "role": "user",
            "content": f"POLICY CONTEXT:\n{context}\n\nQUESTION: {query}{grade_note}\n\nAnswer using ONLY the context above."
        })

        # 🔥 ALWAYS USE CENTRALIZED INVOCATION
        response = invoke_llm(msgs, model_override=model_override)

        answer = response.content.strip()

    except Exception as e:
        logger.error("Generation error: %s", e)
        answer = f"⚠️ Generation error: {e}"

    sources = re.findall(r"Source:\s*(.+)", answer)

    return trace(
        {
            **state,
            "answer": answer,
            "sources": [s.strip() for s in sources]
        },
        node="generate",
        data={
            "len": len(answer),
            "model": model_override or "default"
        }
    )
