"""
Context Assembler — Merges all tool outputs into a deterministically
structured context block that the generator and verifier both parse.
"""

import logging
from typing import Any, List, Optional

from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)
_SEP = "\n" + "─" * 60 + "\n"


def _fmt_sql(rows: List[dict]) -> str:
    if not rows:
        return "(no structured policy records found)"
    return "\n".join(
        "  " + " | ".join(f"{k}: {v}" for k, v in r.items() if v is not None)
        for r in rows
    )


def _fmt_docs(docs: List[str]) -> str:
    if not docs:
        return "(no policy document chunks retrieved)"
    return "\n\n".join(f"[CHUNK {i+1}]\n{d.strip()}" for i, d in enumerate(docs))


def _fmt_compute(summary: str, result: Optional[float]) -> str:
    if result is None:
        return "(no deterministic computation performed)"
    lines = [summary] if summary else []
    lines.append(f"FINAL COMPUTED VALUE: ₹{result:,.2f}")
    return "\n".join(lines)


def run(state: AgentState) -> AgentState:
    query   = state.get("query", "")
    grade   = state.get("employee_grade", "")
    rows    = state.get("sql_result") or []

    # 🔥 BACKWARD + FORWARD COMPATIBILITY LAYER (ADDED)
    docs_structured = state.get("documents") or [
        {"content": d, "source": "legacy"} for d in state.get("retrieval_docs", [])
    ]
    docs = [d.get("content", "") for d in docs_structured]

    result  = state.get("compute_result")
    summary = state.get("compute_summary", "")
    history = state.get("history") or []

    parts = [
        "═══ CORPORATE POLICY CONTEXT ═══",
        f"QUERY: {query}" + (f"  [Grade: {grade}]" if grade else ""),
    ]

    if rows:
        parts += [_SEP + "📋 STRUCTURED POLICY DATA", _fmt_sql(rows)]

    if docs:
        parts += [_SEP + "📄 POLICY DOCUMENT EXCERPTS", _fmt_docs(docs)]

    if result is not None:
        parts += [_SEP + "🔢 DETERMINISTIC COMPUTATION", _fmt_compute(summary, result)]

    if history:
        recent = "\n".join(f"  {m['role'].upper()}: {m['content']}" for m in history[-4:])
        parts += [_SEP + "💬 CONVERSATION CONTEXT", recent]

    context = "\n".join(parts)
    tokens  = len(context) // 4

    logger.info("Context assembled: %d chars (~%d tokens)", len(context), tokens)

    return trace(
        {
            **state,
            "context": context,
            "token_count": tokens,
            # 🆕 optional sync (non-breaking)
            "context_tokens": tokens
        },
        node="context_assembler",
        data={"tokens": tokens},
    )