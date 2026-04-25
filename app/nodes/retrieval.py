"""
Retrieval Node — Semantic vector search over policy document corpus.

- Uses FAISS (default) or Chroma (VECTOR_STORE=chroma)
- MMR reranking to reduce chunk redundancy
- Grade-boosted queries: prepends grade context so embeddings are more precise
"""

import logging
import os

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


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    grade = state.get("employee_grade")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "6"))

    # Boost retrieval precision with grade context
    search_query = f"[Grade: {grade}] {query}" if grade else query

    logger.info("Retrieval — query=%r grade=%s top_k=%d", query, grade, top_k)
    try:
        docs = _get_retriever().retrieve(search_query, top_k=top_k)
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        docs = []

    return trace({**state, "retrieval_docs": docs}, node="retrieval", data={"chunks": len(docs)})
