import logging
import os
from typing import List, Any
from app.state import AgentState
from app.tools.retriever import PolicyRetriever
from app.utils.tracing import trace
from app.core.utils import cross_encoder  # ✅ add rerank here

logger = logging.getLogger(__name__)
_retriever: PolicyRetriever | None = None

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS = 120
DEFAULT_TOP_K = 2
MAX_CONTEXT_CHARS = 1500  # ✅ global cap

def _get_retriever() -> PolicyRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PolicyRetriever()
    return _retriever

def _trim_text(text: str, max_tokens: int) -> str:
    return " ".join(text.split()[:max_tokens])

def run(state: AgentState) -> AgentState:
    query = state.get("query", "")
    grade = state.get("employee_grade")
    history = state.get("history") or []
    vector_memory = state.get("vector_memory")

    top_k = min(int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K)), 3)
    base_query = f"[Grade: {grade}] {query}" if grade else query

    if history:
        history_text = " ".join([m.get("content", "") for m in history])[-MAX_HISTORY_CHARS:]
        enhanced_query = f"{base_query}\nContext:\n{history_text}"
    else:
        enhanced_query = base_query

    try:
        docs = _get_retriever().retrieve(enhanced_query, top_k=top_k)
    except Exception as e:
        logger.error("Retrieval error: %s", e)
        docs = []

    if vector_memory:
        try:
            docs.extend(vector_memory.search(query, k=1)[:1])  # ✅ strict limit
        except Exception:
            pass

    # ✅ rerank before compression
    try:
        reranked = cross_encoder.rank(query, docs)
        docs = [d for d, _ in reranked[:top_k]]
    except Exception:
        pass

    # ✅ compress and cap context
    compressed = [_trim_text(getattr(d, "page_content", str(d)), MAX_CHUNK_TOKENS) for d in docs]
    context = "\n\n".join(compressed)[:MAX_CONTEXT_CHARS]

    return trace(
        {**state, "retrieval_docs": docs, "context": context},
        node="retrieval",
        data={"chunks": len(compressed), "chars": len(context)},
    )
