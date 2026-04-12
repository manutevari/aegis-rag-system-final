# =============================================================================
# AEGIS — RAG PIPELINE ENTRY POINT
# rag_pipeline.py
#
# Public interface:
#   from rag_pipeline import process_query
#   result = process_query("What is the hotel limit for domestic travel?")
#
# This module wraps graph.run_query() with:
#   1. Structured logging (PipelineLogger)
#   2. Fallback handling (FallbackReason)
#   3. Anti-hallucination check (check_hallucination_risk)
#   4. Precision@K computation on every result
#   5. Consistent output schema validated by ProcessResult Pydantic model
#
# __init__.py re-exports process_query so callers can also do:
#   from rag_pipeline import run_query  (backwards-compat)
# =============================================================================

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from fallback_handler import (
    FallbackReason,
    FallbackResponse,
    handle_fallback,
    check_hallucination_risk,
    NOT_FOUND_PHRASE,
    SAFE_FALLBACK_ANSWER,
)
from logger import get_logger, PipelineLogger

log = get_logger(__name__)

__all__ = ["process_query", "ProcessResult", "run_query"]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ProcessResult(BaseModel):
    """
    Validated result returned by process_query().
    Guarantees every consumer gets a known schema.
    """
    answer:         str         = Field(default=SAFE_FALLBACK_ANSWER)
    category:       str | None  = Field(default=None)
    retrieved:      int         = Field(default=0, ge=0)
    sources:        list[dict]  = Field(default_factory=list)
    messages:       list        = Field(default_factory=list)
    cache_hit:      bool        = Field(default=False)
    rrf_applied:    bool        = Field(default=False)
    is_fallback:    bool        = Field(default=False)
    fallback_reason: str | None = Field(default=None)
    hallucination_risk: bool    = Field(default=False)
    halluc_score:   float       = Field(default=0.0, ge=0.0, le=1.0)
    precision_at_k: dict        = Field(
        default_factory=dict,
        description="Precision@1/3/5 computed from returned sources vs query terms",
    )
    latency_s:      float       = Field(default=0.0, ge=0.0)

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        return v.strip() or SAFE_FALLBACK_ANSWER


# ---------------------------------------------------------------------------
# Precision@K helper (pure Python, no API calls)
# ---------------------------------------------------------------------------

def _compute_precision_at_k(
    sources: list[dict],
    query:   str,
    ks:      list[int] = (1, 3, 5),
) -> dict[str, float]:
    """
    Compute Precision@K for K in `ks`.

    A source is considered "relevant" at query time if its chunk_text
    contains at least one non-trivial query token (>3 chars, not a stopword).
    This is a retrieval-time precision estimate — the gold standard version
    (with ground-truth doc IDs) lives in evaluation.py.

    Returns: {"P@1": float, "P@3": float, "P@5": float}
    """
    _STOP = {
        "what", "is", "the", "for", "how", "can", "does", "when",
        "are", "who", "which", "do", "my", "our", "your", "their",
        "this", "that", "with", "about", "have", "will", "would",
    }
    query_tokens = {
        t.lower().strip(".,?!")
        for t in query.split()
        if len(t) > 3 and t.lower() not in _STOP
    }

    if not query_tokens or not sources:
        return {f"P@{k}": 0.0 for k in ks}

    result: dict[str, float] = {}
    for k in ks:
        top_k = sources[:k]
        hits  = sum(
            1 for s in top_k
            if any(tok in s.get("chunk_text", "").lower() for tok in query_tokens)
        )
        result[f"P@{k}"] = round(hits / max(len(top_k), 1), 4)

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_query(
    query:              str,
    top_k:              int   = 25,
    num_queries:        int   = 4,
    use_reranking:      bool  = True,
    use_summarisation:  bool  = False,
    max_context_tokens: int   = 3000,
    tfidf_alpha:        float = 0.15,
    use_cache:          bool  = True,
    halluc_check:       bool  = True,
    halluc_threshold:   float = 0.35,
) -> ProcessResult:
    """
    Full AEGIS RAG pipeline entry point.

    Wraps graph.run_query() with:
      • Structured pipeline logging (PipelineLogger)
      • Input validation + empty-query fallback
      • Anti-hallucination check (configurable threshold)
      • Precision@1/3/5 computation
      • Consistent ProcessResult schema (Pydantic-validated)

    Args:
        query:              User's natural language question.
        top_k:              Dense retrieval candidates per variant.
        num_queries:        Query expansion variant count.
        use_reranking:      Enable cross-encoder reranking.
        use_summarisation:  Compress context with cheap model.
        max_context_tokens: Hard context token budget.
        tfidf_alpha:        TF-IDF calibration weight [0,1].
        use_cache:          Return cached result if available.
        halluc_check:       Run anti-hallucination check on answer.
        halluc_threshold:   Foreign-token ratio that flags hallucination.

    Returns:
        ProcessResult (Pydantic-validated, always safe to unpack).
    """
    plog = PipelineLogger(query)
    t0   = time.perf_counter()

    # ── Input validation ──────────────────────────────────────────────────
    if not query or not query.strip():
        fb = handle_fallback(
            FallbackReason.EMPTY_QUERY, "process_query",
            query=query, detail="Query string is empty or whitespace-only.",
        )
        return ProcessResult(
            answer=fb.answer, is_fallback=True,
            fallback_reason=fb.reason.value,
            messages=[fb.to_tool_message()],
        )

    plog.node_enter("pipeline", top_k=top_k, num_queries=num_queries,
                    use_reranking=use_reranking)

    # ── Run graph pipeline ────────────────────────────────────────────────
    try:
        from graph import run_query as _run_query
        raw = _run_query(
            query=query,
            top_k=top_k,
            num_queries=num_queries,
            use_reranking=use_reranking,
            use_summarisation=use_summarisation,
            max_context_tokens=max_context_tokens,
            tfidf_alpha=tfidf_alpha,
            use_cache=use_cache,
        )
        plog.node_exit("pipeline", sources=len(raw.get("sources", [])))
    except Exception as exc:
        plog.error("pipeline", exc)
        fb = handle_fallback(
            FallbackReason.API_ERROR, "process_query",
            query=query, detail=f"Graph execution failed: {exc}",
        )
        return ProcessResult(
            answer=fb.answer, is_fallback=True,
            fallback_reason=fb.reason.value,
            messages=[fb.to_tool_message()],
        )

    answer  = raw.get("answer", "") or ""
    sources = raw.get("sources", []) or []

    # ── Empty results fallback ────────────────────────────────────────────
    if not sources and NOT_FOUND_PHRASE not in answer:
        plog.fallback("process_query", "No sources retrieved")
        fb = handle_fallback(
            FallbackReason.EMPTY_RESULTS, "process_query",
            query=query,
            detail="Pipeline returned 0 sources. Check that documents are ingested.",
            answer=answer if answer else None,
        )
        is_fallback   = True
        fallback_reason = fb.reason.value
        messages_extra = [fb.to_tool_message()]
    else:
        is_fallback     = False
        fallback_reason = None
        messages_extra  = []

    # ── Anti-hallucination check ──────────────────────────────────────────
    is_risky   = False
    halluc_score = 0.0
    if halluc_check and sources and answer:
        context_texts = [s.get("chunk_text", "") for s in sources]
        is_risky, halluc_score = check_hallucination_risk(
            answer, context_texts, threshold=halluc_threshold
        )
        if is_risky:
            plog.fallback(
                "halluc_check",
                f"Hallucination risk score={halluc_score:.2%}",
                halluc_score=halluc_score,
            )
            fb_halluc = handle_fallback(
                FallbackReason.HALLUCINATION, "node_generate",
                query=query,
                detail=f"Answer contained {halluc_score:.0%} foreign tokens. "
                       f"Replacing with safe fallback.",
                halluc_score=halluc_score,
            )
            # Replace answer with safe text; keep sources for audit
            answer        = fb_halluc.answer
            is_fallback   = True
            fallback_reason = fb_halluc.reason.value
            messages_extra.append(fb_halluc.to_tool_message())

    # ── Precision@K ──────────────────────────────────────────────────────
    pak = _compute_precision_at_k(sources, query)
    plog.metric("precision@1", pak.get("P@1", 0.0))
    plog.metric("precision@3", pak.get("P@3", 0.0))
    plog.metric("precision@5", pak.get("P@5", 0.0))

    latency = round(time.perf_counter() - t0, 3)
    plog.pipeline_done(len(answer), len(sources))

    return ProcessResult(
        answer         = answer,
        category       = raw.get("category"),
        retrieved      = raw.get("retrieved", 0),
        sources        = sources,
        messages       = (raw.get("messages") or []) + messages_extra,
        cache_hit      = raw.get("cache_hit", False),
        rrf_applied    = raw.get("rrf_applied", False),
        is_fallback    = is_fallback,
        fallback_reason= fallback_reason,
        hallucination_risk = is_risky,
        halluc_score   = halluc_score,
        precision_at_k = pak,
        latency_s      = latency,
    )


# ---------------------------------------------------------------------------
# Backwards-compatible alias used by retrieval.py and streamlit_app.py
# ---------------------------------------------------------------------------

def run_query(
    query:              str,
    top_k:              int   = 25,
    num_queries:        int   = 4,
    use_reranking:      bool  = True,
    use_summarisation:  bool  = False,
    max_context_tokens: int   = 3000,
    tfidf_alpha:        float = 0.15,
    use_cache:          bool  = True,
) -> dict[str, Any]:
    """
    Backwards-compatible wrapper.
    Returns a plain dict matching the original graph.run_query() schema,
    plus the new fields (is_fallback, halluc_score, precision_at_k).
    """
    result = process_query(
        query=query,
        top_k=top_k,
        num_queries=num_queries,
        use_reranking=use_reranking,
        use_summarisation=use_summarisation,
        max_context_tokens=max_context_tokens,
        tfidf_alpha=tfidf_alpha,
        use_cache=use_cache,
    )
    return result.model_dump()
