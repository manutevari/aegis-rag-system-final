"""Advanced retrieval node backed by the shared vector store.

The retrieval stage follows the Project Aegis guidelines: query expansion,
HyDE-style policy text, metadata pre-filtering, latest-version post-filtering,
broad retrieval, and top-k reranking before context assembly.
"""

import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.core.stability_patch import safe_get, with_updates
from app.core.vector_store import ensure_vectorstore_ready, get_retriever
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

MAX_HISTORY_CHARS = 500
MAX_CHUNK_TOKENS = 260
DEFAULT_FINAL_K = 5
DEFAULT_BROAD_K = 25
MAX_CONTEXT_CHARS = 6000

CATEGORY_KEYWORDS = {
    "travel": ["travel", "taxi", "cab", "uber", "lyft", "rideshare", "flight", "hotel", "mileage", "fuel", "rental", "airport"],
    "hr": ["leave", "absence", "maternity", "conduct", "discipline", "workplace", "pto", "holiday"],
    "security": ["security", "privacy", "vpn", "data", "incident", "device", "password", "cyber"],
    "training": ["training", "learning", "tuition", "course", "certification"],
    "compensation": ["salary", "bonus", "pay", "compensation", "hra", "performance", "allowance"],
}

QUERY_HINTS = {
    "taxi": ["licensed taxi reimbursement", "ground transportation airport hotel client site", "cab fare corporate travel"],
    "cab": ["licensed taxi reimbursement", "ground transportation expense policy", "rideshare and taxi rules"],
    "uber": ["rideshare reimbursement", "UberX Lyft Standard ground transportation", "airport hotel client location rideshare"],
    "lyft": ["rideshare reimbursement", "UberX Lyft Standard ground transportation", "airport hotel client location rideshare"],
    "allowance": ["employee allowance reimbursement limit", "policy entitlement rate", "eligible expense allowance"],
    "hotel": ["lodging nightly rate threshold", "hotel reimbursement policy", "accommodation limit by city tier"],
    "fuel": ["rental car fuel reimbursement", "personal vehicle mileage fuel restriction", "gas receipt expense policy"],
    "leave": ["leave and absence policy", "paid time off approval process", "employee absence rules"],
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _trim_text(text: str, max_tokens: int) -> str:
    return " ".join((text or "").split()[:max_tokens])


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


def _terms(text: str) -> List[str]:
    return [term for term in TOKEN_RE.findall((text or "").lower()) if len(term) > 2]


def _detect_policy_category(query: str) -> Optional[str]:
    lower = (query or "").lower()
    scores = {
        category: sum(1 for keyword in keywords if keyword in lower)
        for category, keywords in CATEGORY_KEYWORDS.items()
    }
    category, score = max(scores.items(), key=lambda item: item[1])
    return category if score else None


def _metadata_filter(query: str) -> Optional[Dict[str, Any]]:
    category = _detect_policy_category(query)
    if not category:
        return None
    return {"policy_category": category}


def _expand_query(query: str) -> List[str]:
    lower = (query or "").lower()
    variants = [query]
    for keyword, hints in QUERY_HINTS.items():
        if keyword in lower:
            variants.extend(hints)
    variants.extend([f"{query} policy", f"{query} reimbursement rules"])

    seen = set()
    unique = []
    for variant in variants:
        clean = " ".join(str(variant).split())
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            unique.append(clean)
    return unique[:5]


def _hyde_document(query: str, category: Optional[str]) -> str:
    topic = f"{category} policy" if category else "corporate policy"
    return (
        f"A {topic} answer for the question '{query}' would state the exact eligibility, "
        "approved use cases, reimbursement limits, exclusions, documentation, receipt, "
        "approval, and source policy section for the employee expense or benefit."
    )


def _retrieve(query: str, top_k: int, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Any]:
    try:
        retriever = get_retriever(k=top_k, metadata_filter=metadata_filter)
    except TypeError:
        retriever = get_retriever(k=top_k)

    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query) or [])
    if hasattr(retriever, "get_relevant_documents"):
        return list(retriever.get_relevant_documents(query) or [])
    return []


def _dedupe(docs: Iterable[Any]) -> List[Any]:
    seen = set()
    unique = []
    for doc in docs:
        metadata = _metadata(doc)
        text = _content(doc).strip()
        key = (
            metadata.get("source_path") or metadata.get("source") or "unknown",
            metadata.get("chunk_index"),
            text[:300],
        )
        if not text or key in seen:
            continue
        seen.add(key)
        unique.append(doc)
    return unique


def _parse_date(value: Any) -> date:
    if not value or str(value).lower() == "unknown":
        return date.min
    for fmt in ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(str(value), fmt).date()
        except ValueError:
            pass
    return date.min


def _document_family(metadata: Dict[str, Any]) -> str:
    document_id = str(metadata.get("document_id") or metadata.get("source") or "unknown")
    return re.sub(r"[-_]?v\d+$", "", document_id, flags=re.I).lower()


def _post_filter_latest(docs: List[Any]) -> List[Any]:
    latest_by_family: Dict[str, date] = {}
    for doc in docs:
        metadata = _metadata(doc)
        family = _document_family(metadata)
        effective = _parse_date(metadata.get("effective_date"))
        latest_by_family[family] = max(latest_by_family.get(family, date.min), effective)

    filtered = []
    for doc in docs:
        metadata = _metadata(doc)
        family = _document_family(metadata)
        effective = _parse_date(metadata.get("effective_date"))
        if effective == date.min or effective == latest_by_family.get(family, date.min):
            filtered.append(doc)
    return filtered


def _lexical_score(query: str, doc: Any) -> float:
    metadata = _metadata(doc)
    text = " ".join(
        [
            _content(doc),
            str(metadata.get("h1_header", "")),
            str(metadata.get("h2_header", "")),
            str(metadata.get("h3_header", "")),
            str(metadata.get("policy_category", "")),
        ]
    ).lower()
    query_terms = set(_terms(query))
    if not query_terms:
        return 0.0

    text_terms = set(_terms(text))
    overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
    phrase_bonus = 0.25 if query.lower() in text else 0.0
    category = _detect_policy_category(query)
    category_bonus = 0.15 if category and metadata.get("policy_category") == category else 0.0
    table_bonus = 0.05 if metadata.get("contains_table") else 0.0
    return overlap + phrase_bonus + category_bonus + table_bonus


def _cross_encoder_scores(query: str, docs: List[Any]) -> Optional[List[float]]:
    provider = os.getenv("RERANK_PROVIDER", "lexical").strip().lower()
    if provider not in {"cross_encoder", "bge_reranker"}:
        return None

    model_name = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
        pairs = [(query, _content(doc)) for doc in docs]
        return [float(score) for score in model.predict(pairs)]
    except Exception as exc:
        logger.warning("Cross-encoder reranker unavailable; using lexical rerank: %s", exc)
        return None


def _rerank(query: str, docs: List[Any], final_k: int) -> List[Tuple[Any, float]]:
    scores = _cross_encoder_scores(query, docs)
    if scores is None:
        scores = [_lexical_score(query, doc) for doc in docs]

    ranked = sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)
    return ranked[:final_k]


def _serialise(doc: Any, rerank_score: float) -> Dict[str, Any]:
    metadata = _metadata(doc)
    return {
        "content": _trim_text(_content(doc), MAX_CHUNK_TOKENS),
        "source": _source(metadata),
        "metadata": metadata,
        "rerank_score": round(float(rerank_score), 4),
    }


def _context_block(doc: Dict[str, Any]) -> str:
    metadata = doc.get("metadata") or {}
    header = [
        f"Source: {doc.get('source', 'unknown')}",
        f"Document: {metadata.get('document_id', 'unknown')}",
        f"Section: {metadata.get('section_path') or metadata.get('h2_header') or metadata.get('h1_header') or 'unknown'}",
        f"Effective date: {metadata.get('effective_date', 'unknown')}",
        f"Relevance score: {doc.get('rerank_score', 0)}",
    ]
    return "\n".join(header + ["", doc.get("content", "")])


def run(state: AgentState) -> AgentState:
    query = safe_get(state, "query", "") or ""
    grade = safe_get(state, "employee_grade")
    history = safe_get(state, "history") or []
    vector_memory = safe_get(state, "vector_memory")

    final_k = min(max(int(os.getenv("RERANK_TOP_K", DEFAULT_FINAL_K)), 1), 10)
    broad_k = min(max(int(os.getenv("RETRIEVAL_BROAD_K", DEFAULT_BROAD_K)), final_k), 50)
    metadata_filter = _metadata_filter(query)
    category = metadata_filter.get("policy_category") if metadata_filter else None

    base_query = f"[Grade: {grade}] {query}" if grade else query
    if history:
        history_text = " ".join([m.get("content", "") for m in history])[-MAX_HISTORY_CHARS:]
        base_query = f"{base_query}\nConversation context:\n{history_text}"

    variants = _expand_query(query)
    hyde = _hyde_document(query, category)
    search_queries = [base_query] + variants + [hyde]

    raw_docs: List[Any] = []
    retrieval_error = ""
    filter_fallback = False

    try:
        ensure_vectorstore_ready(auto_ingest=os.getenv("AUTO_INGEST", "true").lower() != "false")
        for search_query in search_queries:
            raw_docs.extend(_retrieve(search_query, top_k=broad_k, metadata_filter=metadata_filter))
        if not raw_docs and metadata_filter:
            filter_fallback = True
            for search_query in search_queries:
                raw_docs.extend(_retrieve(search_query, top_k=broad_k, metadata_filter=None))
    except Exception as exc:
        retrieval_error = str(exc)
        logger.error("Retrieval error: %s", exc, exc_info=True)

    if vector_memory:
        try:
            raw_docs.extend(vector_memory.search(query, k=1)[:1])
        except Exception:
            pass

    pooled = _post_filter_latest(_dedupe(raw_docs))
    ranked = _rerank(query, pooled, final_k=final_k) if pooled else []
    docs = [_serialise(doc, score) for doc, score in ranked]
    context = "\n\n---\n\n".join(_context_block(doc) for doc in docs)[:MAX_CONTEXT_CHARS]

    updates = {
        "documents": docs,
        "retrieval_docs": [doc["content"] for doc in docs],
        "context": context,
        "query_expansions": variants,
        "hyde_query": hyde,
        "metadata_filter": metadata_filter or {},
        "retrieval_diagnostics": {
            "broad_k": broad_k,
            "final_k": final_k,
            "pooled_chunks": len(pooled),
            "filter_fallback": filter_fallback,
            "reranker": os.getenv("RERANK_PROVIDER", "lexical"),
        },
    }
    if retrieval_error:
        updates["error"] = retrieval_error

    return trace(
        with_updates(state, **updates),
        node="retrieval",
        data={
            "chunks": len(docs),
            "pooled_chunks": len(pooled),
            "chars": len(context),
            "filter": metadata_filter or {},
            "expanded_queries": variants,
            "filter_fallback": filter_fallback,
            "error": retrieval_error,
        },
    )
