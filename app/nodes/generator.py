"""
Generator Node — Strict, grounded LLM answer generation.
Uses ONLY the assembled context. Never invents numbers.
"""
import logging, os, re
from typing import List
from app.state import AgentState
from app.utils.tracing import trace

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

    grade_note = f" (Employee grade: {grade})" if grade else ""

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        )
        msgs = [{"role": "system", "content": _SYSTEM}]
        msgs += history[-4:]
        msgs.append({"role": "user", "content":
            f"POLICY CONTEXT:\n{context}\n\nQUESTION: {query}{grade_note}\n\nAnswer using ONLY the context above."})
        answer = llm.invoke(msgs).content.strip()
    except Exception as e:
        logger.error("Generation error: %s", e)
        answer = f"⚠️ Generation error: {e}"

    sources = re.findall(r"Source:\s*(.+)", answer)
    return trace({**state, "answer": answer, "sources": [s.strip() for s in sources]},
                 node="generate", data={"len": len(answer)})
