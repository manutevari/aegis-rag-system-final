# =============================================================================
# AEGIS — LANGGRAPH PIPELINE
# graph.py
#
# Wires all retrieval_nodes into a compiled LangGraph StateGraph.
#
# Graph topology (linear — no conditional branching needed):
#
#   START
#     │
#     ▼
#   router          ← classify intent (keyword-first, LLM fallback)
#     │
#     ▼
#   expand_query    ← multi-query expansion (3 variants)
#     │
#     ▼
#   hyde            ← hypothetical document embedding
#     │
#     ▼
#   retrieve        ← broad retrieval Top-25 × N variants
#     │
#     ▼
#   post_filter     ← drop stale policy versions
#     │
#     ▼
#   rerank          ← cross-encoder Top-25 → Top-5
#     │
#     ▼
#   generate        ← stuff-documents chain → final answer
#     │
#     ▼
#   END
#
# Usage:
#   from graph import build_graph, run_query
#
#   graph  = build_graph()
#   result = run_query("Can I expense a taxi?")
#   print(result["final_answer"])
#   for msg in result["messages"]:
#       print(type(msg).__name__, msg.content[:80])
# =============================================================================

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.graph import END, START, StateGraph

from graph_state import AegisState, HumanMessage, SystemMessage
from cache import get_cache
from retrieval_nodes import (
    node_expand_query,
    node_generate,
    node_hyde,
    node_post_filter,
    node_rerank,
    node_retrieve,
    node_router,
)

# ---------------------------------------------------------------------------
# Pipeline entry SystemMessage — injected once at graph start
# ---------------------------------------------------------------------------

_PIPELINE_SYSTEM = (
    "AEGIS Enterprise RAG — Retrieval Pipeline\n"
    "Steps: intent_routing → query_expansion → HyDE → "
    "broad_retrieval → post_filter → cross_encoder_rerank → answer_generation\n"
    "All decisions are logged as ToolMessage entries in this message thread."
)


# ---------------------------------------------------------------------------
# Input node — validates and seeds AegisState from raw query string
# ---------------------------------------------------------------------------

def node_input(state: AegisState) -> dict:
    """
    Seed the pipeline with a SystemMessage describing the run,
    and a HumanMessage carrying the user's query.
    Both are appended to messages for a complete audit trail.
    """
    return {
        "messages": [
            SystemMessage(content=_PIPELINE_SYSTEM),
            HumanMessage(content=state.query),
        ]
    }


# ---------------------------------------------------------------------------
# Build + compile the graph
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def build_graph():
    """
    Construct and compile the Aegis LangGraph StateGraph.
    Cached — safe to call repeatedly; returns the same compiled graph.
    """
    builder = StateGraph(AegisState)

    # Register nodes
    builder.add_node("input",        node_input)
    builder.add_node("router",       node_router)
    builder.add_node("expand_query", node_expand_query)
    builder.add_node("hyde",         node_hyde)
    builder.add_node("retrieve",     node_retrieve)
    builder.add_node("post_filter",  node_post_filter)
    builder.add_node("rerank",       node_rerank)
    builder.add_node("generate",     node_generate)

    # Wire edges (linear pipeline)
    builder.add_edge(START,          "input")
    builder.add_edge("input",        "router")
    builder.add_edge("router",       "expand_query")
    builder.add_edge("expand_query", "hyde")
    builder.add_edge("hyde",         "retrieve")
    builder.add_edge("retrieve",     "post_filter")
    builder.add_edge("post_filter",  "rerank")
    builder.add_edge("rerank",       "generate")
    builder.add_edge("generate",     END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public entry point
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
    Execute the full Aegis LangGraph pipeline for a user query.

    Args:
        query:              Raw user question string.
        top_k:              Broad retrieval top-K per variant (lecture: topK).
        num_queries:        Query expansion variant count (lecture: numQueries).
        use_reranking:      Enable cross-encoder reranking (A/B test flag).
        use_summarisation:  Compress context with cheap model before generation.
        max_context_tokens: Hard token budget for context window.
        tfidf_alpha:        TF-IDF calibration blend weight [0,1].
        use_cache:          Return cached result if available (lecture: caching).

    Returns:
        dict with keys:
          answer      — grounded LLM answer
          category    — detected policy category (or None)
          retrieved   — number of broad retrieval results (post-filter)
          sources     — list of reranked ChunkResult dicts
          messages    — full audit trail (System/Human/AI/Tool messages)
          cache_hit   — True if result was served from cache
    """
    # ── Cache lookup (lecture: "caching layers to reduce latency/cost") ──
    cache     = get_cache()
    cache_key = cache.make_key(query, alpha=tfidf_alpha)
    if use_cache:
        hit = cache.get(cache_key)
        if hit is not None:
            hit["cache_hit"] = True
            return hit

    graph = build_graph()

    # Pydantic validates all config fields on construction
    initial_state = AegisState(
        query=query,
        top_k=top_k,
        num_queries=num_queries,
        use_reranking=use_reranking,
        use_summarisation=use_summarisation,
        max_context_tokens=max_context_tokens,
        tfidf_calibration_alpha=tfidf_alpha,
    )
    final_state = graph.invoke(initial_state)

    sources = [
        c.model_dump() if hasattr(c, "model_dump") else c
        for c in (final_state.get("reranked_chunks") or [])
    ]

    result = {
        "answer":    final_state.get("final_answer", ""),
        "category":  final_state.get("detected_category"),
        "retrieved": len(final_state.get("broad_results") or []),
        "sources":   sources,
        "messages":  final_state.get("messages") or [],
        "cache_hit": False,
        "rrf_applied": final_state.get("rrf_applied", False),
    }

    # ── Cache store ───────────────────────────────────────────────────────
    if use_cache:
        # Store without messages (not JSON-serialisable as-is)
        cacheable = {k: v for k, v in result.items() if k != "messages"}
        cache.set(cache_key, query, cacheable)

    return result
