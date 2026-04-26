"""
AgentState — Single source of truth passed between every LangGraph node.
Backward-compatible + research-grade extensions (non-breaking).
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):

    # ── Input ──────────────────────────────────────────────────────────────
    query: str
    employee_grade: Optional[str]
    history: List[Dict[str, str]]

    # ── Routing ────────────────────────────────────────────────────────────
    route: str                         # "sql" | "retrieval" | "compute" | "direct"
    needs_compute: bool

    # 🆕 Research extension (SAFE)
    max_retries: int

    # ── Tool outputs ───────────────────────────────────────────────────────
    sql_result: Optional[List[dict]]
    sql_params: Optional[dict]

    retrieval_docs: List[str]          # existing (DO NOT CHANGE)

    # 🆕 Research-compatible structured docs (optional, non-breaking)
    documents: List[Dict[str, Any]]    # [{content, score, source}]
    retrieved: bool

    compute_result: Optional[float]
    compute_steps: List[str]
    compute_summary: str

    # ── Context ────────────────────────────────────────────────────────────
    context: str
    token_count: int
    context_summarized: bool

    # 🆕 alias for research (non-breaking)
    context_tokens: int

    # ── Generation ─────────────────────────────────────────────────────────
    answer: str
    sources: List[str]
    _encrypted_answer: Optional[bytes]

    # ── Verification ───────────────────────────────────────────────────────
    verified: bool
    verification_issues: List[str]
    retry_count: int

    # 🆕 Research additions
    confidence: float

    # ── HITL ───────────────────────────────────────────────────────────────
    hitl_decision: str                 # "approve" | "edit" | "reject"
    hitl_edited_answer: Optional[str]

    # ── Tracing / Observability ────────────────────────────────────────────
    trace_log: List[Dict[str, Any]]

    error: Optional[str]

    # 🆕 Research-safe error aggregation
    errors: List[str]