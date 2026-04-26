import logging
import os
from typing import List, Any

from app.state import AgentState
from app.tools.retriever import PolicyRetriever
from app.utils.tracing import trace

logger = logging.getLogger(__name__)
_retriever: PolicyRetriever | None = None

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS = 120
DEFAULT_TOP_K = 2


def _get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever


def _deduplicate_docs(docs: List[Any]) -> List[Any]:
    seen = set()
    unique_docs = []
    for d in docs:
        content = getattr(d, "page_content", str(d))
        if content not in seen:
            seen.add(content)
            unique_docs.append(d)
    return unique_docs


def _trim_text(text: str, max_tokens: int) -> str:
    words = text.split()
    return " ".join(words[:max_tokens])


def _compress_docs(docs: List[Any], k: int) -> List[str]:
    texts = []
    for d in docs[:k]:
        content = getattr(d, "page_content", str(d))
        texts.append(_trim_text(content, MAX_CHUNK_TOKENS))
    return texts


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    grade = state.get("employee_grade")
    history = state.get("history") or []
    vector_memory = state.get("vector_memory")

    top_k = int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K))
    top_k = min(top_k, 3)

    base_query = f"[Grade: {grade}] {query}" if grade else query

    if history:
        history_text = " ".join([m.get("content", "") for m in history])
        history_trimmed = history_text[-MAX_HISTORY_CHARS:]
        enhanced_query = f"{base_query}\nContext:\n{history_trimmed}"
    else:
        enhanced_query = base_query

    try:
        docs = _get_retriever().retrieve(enhanced_query, top_k=top_k)
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        docs = []

    if vector_memory:
        try:
            memory_docs = vector_memory.search(query, k=1)
            docs.extend(memory_docs)
        except Exception:
            pass

    docs = _deduplicate_docs(docs)
    compressed_chunks = _compress_docs(docs, k=top_k)

    return trace(
        {
            **state,
            "retrieval_docs": docs,
            "context": "\n\n".join(compressed_chunks),  # ✅ FIXED
        },
        node="retrieval",
        data={"chunks": len(compressed_chunks)},
    )
