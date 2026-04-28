"""LangGraph retrieval node backed by the shared vector store."""

import logging
import os
from typing import Any, Dict, List

from app.core.stability_patch import safe_get, with_updates
from app.core.vector_store import ensure_vectorstore_ready, get_retriever
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS = 180
DEFAULT_TOP_K = 5
MAX_CONTEXT_CHARS = 4000


def _trim_text(text: str, max_tokens: int) -> str:
    return " ".join((text or "").split()[:max_tokens])


def _retrieve(query: str, top_k: int) -> List[Any]:
    ensure_vectorstore_ready(auto_ingest=os.getenv("AUTO_INGEST", "true").lower() != "false")
    retriever = get_retriever(k=top_k)

    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query) or [])
    if hasattr(retriever, "get_relevant_documents"):
        return list(retriever.get_relevant_documents(query) or [])
    return []


def _content(doc: Any) -> str:
    if isinstance(doc, dict):
        return str(doc.get("content") or doc.get("page_content") or "")
    return str(getattr(doc, "page_content", doc))


def _metadata(doc: Any) -> Dict[str, Any]:
    if isinstance(doc, dict):
        return dict(doc.get("metadata") or {})
    return dict(getattr(doc, "metadata", {}) or {})


def _source(metadata: Dict[str, Any]) -> str:
    return str(
        metadata.get("source")
        or metadata.get("source_file")
        or metadata.get("source_path")
        or "unknown"
    )


def _serialise(doc: Any) -> Dict[str, Any]:
    metadata = _metadata(doc)
    return {
        "content": _trim_text(_content(doc), MAX_CHUNK_TOKENS),
        "source": _source(metadata),
        "metadata": metadata,
    }


def _dedupe(docs: List[Any]) -> List[Any]:
    seen = set()
    unique = []
    for doc in docs:
        text = _content(doc).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(doc)
    return unique


def run(state: AgentState) -> AgentState:
    query = safe_get(state, "query", "") or ""
    grade = safe_get(state, "employee_grade")
    history = safe_get(state, "history") or []
    vector_memory = safe_get(state, "vector_memory")

    top_k = min(max(int(os.getenv("RETRIEVAL_TOP_K", DEFAULT_TOP_K)), 1), 8)
    base_query = f"[Grade: {grade}] {query}" if grade else query

    if history:
        history_text = " ".join([m.get("content", "") for m in history])[-MAX_HISTORY_CHARS:]
        enhanced_query = f"{base_query}\nContext:\n{history_text}"
    else:
        enhanced_query = base_query

    raw_docs: List[Any] = []
    retrieval_error = ""

    try:
        raw_docs = _retrieve(enhanced_query, top_k=top_k)
    except Exception as exc:
        retrieval_error = str(exc)
        logger.error("Retrieval error: %s", exc, exc_info=True)

    if vector_memory:
        try:
            raw_docs.extend(vector_memory.search(query, k=1)[:1])
        except Exception:
            pass

    docs = [_serialise(doc) for doc in _dedupe(raw_docs)]
    context = "\n\n".join(doc["content"] for doc in docs)[:MAX_CONTEXT_CHARS]

    updates = {
        "documents": docs,
        "retrieval_docs": [doc["content"] for doc in docs],
        "context": context,
    }
    if retrieval_error:
        updates["error"] = retrieval_error

    return trace(
        with_updates(state, **updates),
        node="retrieval",
        data={"chunks": len(docs), "chars": len(context), "error": retrieval_error},
    )
