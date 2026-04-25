"""
AgentState — Single source of truth passed between every LangGraph node.
All fields use total=False so nodes only need to return what they changed.
"""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────
    query: str                         # Raw employee query
    employee_grade: Optional[str]      # Detected or explicit grade: L1-L7, VP, SVP
    history: List[Dict[str, str]]      # Conversation turns [{"role":…,"content":…}]

    # ── Routing ────────────────────────────────────────────────────────────
    route: str                         # "sql" | "retrieval" | "compute" | "direct"
    needs_compute: bool

    # ── Tool outputs ───────────────────────────────────────────────────────
    sql_result: Optional[List[dict]]
    sql_params: Optional[dict]
    retrieval_docs: List[str]
    compute_result: Optional[float]
    compute_steps: List[str]
    compute_summary: str
    context: str                       # Assembled context fed to LLM
    token_count: int
    context_summarized: bool

    # ── Generation ─────────────────────────────────────────────────────────
    answer: str
    sources: List[str]
    _encrypted_answer: Optional[bytes]

    # ── Verification ───────────────────────────────────────────────────────
    verified: bool
    verification_issues: List[str]
    retry_count: int

    # ── HITL ───────────────────────────────────────────────────────────────
    hitl_decision: str                 # "approve" | "edit" | "reject"
    hitl_edited_answer: Optional[str]

    # ── Tracing ────────────────────────────────────────────────────────────
    trace_log: List[Dict[str, Any]]
    error: Optional[str]
