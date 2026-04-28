"""Token manager with deterministic local context compression."""

import logging
import os
import re

from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD", "3000"))
MAX_SUMMARY_CHARS = int(os.getenv("MAX_SUMMARY_CHARS", "6000"))


def run(state: AgentState) -> AgentState:
    return trace(state, node="token_check", data={"tokens": state.get("token_count", 0)})


def _score_line(line: str, query_terms: set) -> int:
    lower = line.lower()
    score = sum(1 for term in query_terms if term in lower)
    if re.search(r"\d", line):
        score += 2
    if "policy" in lower or "source" in lower:
        score += 1
    return score


def summarize(state: AgentState) -> AgentState:
    context = state.get("context", "") or ""
    query = state.get("query", "") or ""
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}

    lines = [line.strip() for line in context.splitlines() if line.strip()]
    ranked = sorted(lines, key=lambda line: _score_line(line, query_terms), reverse=True)
    compressed_lines = ranked[:80]
    compressed = "\n".join(compressed_lines)[:MAX_SUMMARY_CHARS]

    if not compressed:
        compressed = context[:MAX_SUMMARY_CHARS]

    new_tokens = len(compressed) // 4
    return trace(
        {
            **state,
            "context": compressed,
            "token_count": new_tokens,
            "context_summarized": True,
        },
        node="summarize_context",
        data={"new_tokens": new_tokens},
    )
