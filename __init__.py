# =============================================================================
# AEGIS — PACKAGE INITIALISER
# __init__.py
#
# Makes aegis_v4/ a proper Python package.
# Exposes the canonical public API so callers can do:
#
#   from rag_pipeline import process_query     (preferred — full ProcessResult)
#   from rag_pipeline import run_query         (legacy dict-returning alias)
#
# The user-requested pattern:
#   from rag_pipeline import process_query
#   def run_query(query: str):
#       result = process_query(query)
#       return result
#
# is implemented exactly in rag_pipeline.py and re-exported here.
# =============================================================================

from __future__ import annotations

# Primary entry points
from rag_pipeline import process_query, run_query, ProcessResult   # noqa: F401

# Graph-level run (direct, no logging/halluc-check wrapper)
from graph import run_query as _graph_run_query                    # noqa: F401
from graph import build_graph                                       # noqa: F401

# State schema
from graph_state import (                                           # noqa: F401
    AegisState,
    ChunkResult,
    VALID_CATEGORIES,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    tool_log,
)

# Evaluation
from evaluation import (                                            # noqa: F401
    run_eval,
    ab_test,
    DEFAULT_TRUTH_SET,
    SAMPLE_TEST_QUERIES,
    TruthEntry,
    EvalReport,
)

# Utilities
from fallback_handler import (                                      # noqa: F401
    FallbackReason,
    FallbackResponse,
    handle_fallback,
    check_hallucination_risk,
    NOT_FOUND_PHRASE,
    SAFE_FALLBACK_ANSWER,
)
from logger import get_logger, PipelineLogger                       # noqa: F401
from cache import get_cache, AegisCache                            # noqa: F401

__version__ = "4.0.0"
__all__ = [
    # Pipeline
    "process_query", "run_query", "ProcessResult",
    "build_graph",
    # State
    "AegisState", "ChunkResult", "VALID_CATEGORIES",
    "SystemMessage", "HumanMessage", "AIMessage", "ToolMessage", "tool_log",
    # Evaluation
    "run_eval", "ab_test", "DEFAULT_TRUTH_SET", "SAMPLE_TEST_QUERIES",
    "TruthEntry", "EvalReport",
    # Fallback
    "FallbackReason", "FallbackResponse", "handle_fallback",
    "check_hallucination_risk", "NOT_FOUND_PHRASE", "SAFE_FALLBACK_ANSWER",
    # Logging
    "get_logger", "PipelineLogger",
    # Cache
    "get_cache", "AegisCache",
]
