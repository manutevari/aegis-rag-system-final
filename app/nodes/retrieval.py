"""
Retrieval Node — Semantic vector search over policy document corpus.

Enhancements (non-breaking):
- History-aware query enrichment
- Long-term vector memory recall
- Deduplication of chunks
- Safe fallbacks + logging
"""

import logging
import os
from typing import List, Any

from app.state import AgentState
from app.tools.retriever import PolicyRetriever
from app.utils.tracing import trace

logger = logging.getLogger(__name__)
_retriever: PolicyRetriever | None = None


def _get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


def _deduplicate_docs(docs: List[Any]) -> List[Any]:
    """Remove duplicate documents based on content (safe for LangChain docs)."""
    seen = set()
    unique_docs = []
    for d in docs:
        content = getattr(d, "page_content", str(d))
        if content not in seen:
            seen.add(content)
            unique_docs.append(d)
    return unique_docs


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    grade = state.get("employee_grade")
    history = state.get("history", "")
    vector_memory = state.get("vector_memory")

    top_k = int(os.getenv("RETRIEVAL_TOP_K", "6"))

    # --- STEP 1: Grade-boosted query ---
    base_query = f"[Grade: {grade}] {query}" if grade else query

    # --- STEP 2: History-aware enrichment (truncate to avoid token explosion) ---
    if history:
        history_trimmed = history[-1000:]  # safe truncation
        enhanced_query = f"{base_query}\nContext:\n{history_trimmed}"
    else:
        enhanced_query = base_query

    logger.info(
        "Retrieval — base_query=%r enhanced_query_len=%d top_k=%d",
        base_query,
        len(enhanced_query),
        top_k,
    )

    # --- STEP 3: Primary retrieval (PolicyRetriever handles FAISS/Chroma + MMR) ---
    try:
        docs = _get_retriever().retrieve(enhanced_query, top_k=top_k)
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        docs = []

    # --- STEP 4: Long-term memory augmentation ---
    if vector_memory:
        try:
            memory_docs = vector_memory.search(query, k=3)
            logger.info("Memory recall — %d chunks", len(memory_docs))
            docs.extend(memory_docs)
        except Exception as e:
            logger.warning("Memory retrieval failed: %s", e)

    # --- STEP 5: Deduplicate results ---
    docs = _deduplicate_docs(docs)

    # --- STEP 6: Final trace output ---
    return trace(
        {
            **state,
            "retrieval_docs": docs,
        },
        node="retrieval",
        data={
            "chunks": len(docs),
            "used_history": bool(history),
            "used_memory": bool(vector_memory),
        },
    )