"""
Generator Node — Strict, grounded LLM answer generation.
"""

import logging, os, re, time
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


# ✅ BACKOFF WRAPPER (uses your invoke_llm)
def safe_invoke_llm(messages, model_override=None, retries=3):
    last_error = None

    for i in range(retries):
        try:
            return invoke_llm(messages, model_override=model_override)

        except Exception as e:
            err = str(e)
            last_error = err

            # 🔥 Handle rate limit properly
            if "429" in err or "quota" in err.lower():
                wait = 2 ** i
                logger.warning(f"[LLM] Rate limit hit. Retry in {wait}s...")
                time.sleep(wait)
                continue

            # ❌ Don't retry other errors
            logger.error(f"[LLM] Non-retryable error: {err}")
            break

    raise Exception(f"LLM failed after retries: {last_error}")


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

        # ✅ USE SAFE WRAPPER (THIS IS THE FIX)
        response = safe_invoke_llm(msgs, model_override=model_override)

        # 🔥 Handle different response formats safely
        answer = getattr(response, "content", None)
        if not answer:
            # fallback if OpenAI-style response
            try:
                answer = response.choices[0].message.content
            except Exception:
                answer = str(response)

        answer = answer.strip()

    except Exception as e:
        logger.error("Generation error: %s", e)
        answer = "⚠️ System busy or rate-limited. Please try again."

    sources = re.findall(r"Source:\s*(.+)", answer)

    return trace(
        {
            **state,
            "answer": answer,
            "sources": [s.strip() for s in sources],
            "retry": False  # 🔥 CRITICAL: stop infinite retry loop
        },
        node="generate",
        data={
            "len": len(answer),
            "model": model_override or "default"
        }
    )
