# =============================================================================
# AEGIS — RETRIEVAL (backwards-compatible shim)
# retrieval.py
#
# All logic now lives in retrieval_nodes.py (individual LangGraph nodes)
# and graph.py (compiled StateGraph pipeline).
#
# This module re-exports the public API that streamlit_app.py and any
# external callers expect, delegating to the new graph-based pipeline.
# =============================================================================

from __future__ import annotations

from graph import run_query                                    # noqa: F401  (re-export)
from retrieval_nodes import (                                  # noqa: F401
    node_router        as _node_router,
    node_expand_query  as _node_expand_query,
    node_hyde          as _node_hyde,
    node_retrieve      as _node_retrieve,
    node_post_filter   as _node_post_filter,
    node_rerank        as _node_rerank,
    node_generate      as _node_generate,
    BROAD_K,
    FINAL_K,
    KEYWORD_MAP,
)
from graph_state import VALID_CATEGORIES                       # noqa: F401

# ---------------------------------------------------------------------------
# Legacy function aliases (used by older test suites / notebooks)
# ---------------------------------------------------------------------------

def detect_category(query: str) -> str | None:
    """Legacy alias — use retrieval_nodes.node_router() in graph context."""
    from graph_state import AegisState
    result = _node_router(AegisState(query=query))
    return result.get("detected_category")


classify_intent = detect_category   # alias


def expand_query(query: str) -> list[str]:
    """Legacy alias."""
    from graph_state import AegisState
    result = _node_expand_query(AegisState(query=query))
    return result.get("query_variants", [query])


def hyde(query: str) -> str:
    """Legacy alias."""
    from graph_state import AegisState
    result = _node_hyde(AegisState(query=query))
    return result.get("hyde_document", query)


def final_pipeline(query: str, index=None) -> str:
    """Legacy alias — returns raw context string."""
    result = run_query(query)
    return "\n\n".join(
        s.get("chunk_text", "") for s in result.get("sources", [])
    )
