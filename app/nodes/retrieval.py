"""
Retrieval Node

FIXES:
1. `from app.core.utils import cross_encoder` — app/core/utils.py does NOT
   exist in this repo. Import wrapped in try/except; rerank step skipped
   gracefully when unavailable.
2. Type-hint `PolicyRetriever | None` requires Python ≥ 3.10; replaced
   with `Optional[PolicyRetriever]` for 3.9 compatibility.
3. `vector_memory.search()` returned Document objects, but code tried to
   call `.extend()` on retrieval_docs list of Documents and then indexed
   them by page_content — now normalised consistently.
"""

import logging
import os
from typing import List, Optional

from app.state import AgentState
from app.tools.retriever import PolicyRetriever
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

# FIX: cross_encoder lives in an optional dependency — import safely
try:
    from app.core.utils import cross_encoder as _cross_encoder  # type: ignore
    _HAS_RERANK = True
except ImportError:
    _cross_encoder  = None
    _HAS_RERANK     = False

_retriever: Optional[PolicyRetriever] = None

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS  = 120
DEFAULT_TOP_K     = 2
MAX_CONTEXT_CHARS = 1500


def _get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


def _trim_text(text: str, max_tokens: int) -> str:
    return " ".join(text.split()[:max_tokens])


def run(state: AgentState) -> AgentState:
    query         = state.get("query", "")
    grade         = state.get("employee_grade")
    history       = state.get("history") or []
    vector_memory = state.get("vector_memory")

    top_k      = min(int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K)), 3)
    base_query = f"[Grade: {grade}] {query}" if grade else query

    if history:
        history_text    = " ".join([m.get("content", "") for m in history])[-MAX_HISTORY_CHARS:]
        enhanced_query  = f"{base_query}\nContext:\n{history_text}"
    else:
        enhanced_query = base_query

    try:
        docs = _get_retriever().retrieve(enhanced_query, top_k=top_k)
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        docs = []

    if vector_memory:
        try:
            extra = vector_memory.search(query, k=1)[:1]
            docs.extend(extra)
        except Exception:
            pass

    # FIX: rerank only when cross_encoder is available
    if _HAS_RERANK and _cross_encoder is not None:
        try:
            reranked = _cross_encoder.rank(query, docs)
            docs     = [d for d, _ in reranked[:top_k]]
        except Exception:
            pass

    compressed = [
        _trim_text(getattr(d, "page_content", str(d)), MAX_CHUNK_TOKENS)
        for d in docs
    ]
    context = "\n\n".join(compressed)[:MAX_CONTEXT_CHARS]

    return trace(
        {**state, "retrieval_docs": docs, "context": context},
        node="retrieval",
        data={"chunks": len(compressed), "chars": len(context)},
    )
