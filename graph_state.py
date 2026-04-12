# =============================================================================
# AEGIS — LANGGRAPH STATE SCHEMA
# graph_state.py
#
# Single source of truth for the typed, Pydantic-validated graph state.
# Every node reads from and writes back to AegisState.
#
# Message roles used throughout:
#   SystemMessage  — pipeline-level instructions / prompts
#   HumanMessage   — user query (one per run)
#   AIMessage      — LLM-generated text (expansions, HyDE, answer)
#   ToolMessage    — structured audit log: WHY a tool was called, WHAT it returned
#
# Pydantic enforcement:
#   • All fields carry validators; malformed updates raise ValidationError
#     before they can corrupt the graph.
#   • Each node returns a PARTIAL dict — LangGraph merges via add_messages
#     (annotated field) or last-write for plain fields.
# =============================================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Re-export message types so the rest of the project imports from one place
# ---------------------------------------------------------------------------
__all__ = [
    "AegisState",
    "ChunkResult",
    "SystemMessage",
    "HumanMessage",
    "AIMessage",
    "ToolMessage",
    "tool_log",
    "VALID_CATEGORIES",
]

VALID_CATEGORIES = frozenset({"Travel", "HR", "Finance", "Legal", "IT", "General"})


# ---------------------------------------------------------------------------
# Helper: build a ToolMessage audit entry
# ---------------------------------------------------------------------------

def tool_log(
    tool_name: str,
    reason: str,
    inputs: dict[str, Any],
    outputs: Any,
    tool_call_id: str | None = None,
) -> ToolMessage:
    """
    Construct a ToolMessage that records WHY a tool was called and WHAT it returned.
    Logged into AegisState.messages so the full decision trail is auditable.

    Args:
        tool_name:    Logical name of the operation (e.g. "expand_query").
        reason:       One-sentence explanation of why this step was triggered.
        inputs:       Inputs passed to the tool (serialisable dict).
        outputs:      Raw return value (will be JSON-encoded).
        tool_call_id: Stable ID; auto-generated from tool_name + timestamp if omitted.
    """
    call_id = tool_call_id or f"{tool_name}_{datetime.utcnow().strftime('%H%M%S%f')}"
    content = json.dumps(
        {
            "tool":    tool_name,
            "reason":  reason,
            "inputs":  inputs,
            "outputs": outputs,
        },
        default=str,    # handles dates, Pydantic models, etc.
        ensure_ascii=False,
    )
    return ToolMessage(content=content, tool_call_id=call_id)


# ---------------------------------------------------------------------------
# ChunkResult — typed wrapper for a single retrieved + reranked chunk
# ---------------------------------------------------------------------------

class ChunkResult(BaseModel):
    """Validated wrapper for one retrieved policy chunk."""

    chunk_id:        str   = Field(..., description="Pinecone vector ID")
    document_id:     str   = Field(..., description="Source document identifier")
    policy_category: str   = Field(..., description="Travel|HR|Finance|Legal|IT|General")
    policy_owner:    str   = Field(default="Unknown")
    effective_date:  str   = Field(default="")
    h1_header:       str   = Field(default="")
    h2_header:       str   = Field(default="")
    chunk_text:      str   = Field(..., description="Raw chunk content")
    is_table:        bool  = Field(default=False)
    vector_score:    float = Field(default=0.0, ge=0.0)
    rerank_score:    float = Field(default=0.0)

    @field_validator("policy_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        return v if v in VALID_CATEGORIES else "General"

    @field_validator("effective_date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Accept YYYY-MM-DD or empty; normalise everything else to empty."""
        import re
        return v if re.match(r"^\d{4}-\d{2}-\d{2}$", v) else ""

    @classmethod
    def from_pinecone_match(cls, match: dict) -> "ChunkResult":
        meta = match.get("metadata", {})
        return cls(
            chunk_id=match.get("id", "unknown"),
            document_id=meta.get("document_id", "unknown"),
            policy_category=meta.get("policy_category", "General"),
            policy_owner=meta.get("policy_owner", "Unknown"),
            effective_date=meta.get("effective_date", ""),
            h1_header=meta.get("h1_header", ""),
            h2_header=meta.get("h2_header", ""),
            chunk_text=meta.get("chunk_text", ""),
            is_table=bool(meta.get("is_table", False)),
            vector_score=float(match.get("score", 0.0)),
            rerank_score=float(match.get("rerank_score", 0.0)),
        )


# ---------------------------------------------------------------------------
# AegisState — the single graph state object
# ---------------------------------------------------------------------------

class AegisState(BaseModel):
    """
    Pydantic-validated LangGraph state for the Aegis RAG pipeline.

    Field update rules (enforced by LangGraph):
      • messages   — accumulated with add_messages (append-only)
      • all others — last-write-wins (node returns partial dict)

    Every node MUST return a dict whose keys are a subset of this model's fields.
    Unknown keys raise a ValidationError before the state is updated.
    """

    # ── Conversation / audit log ──────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="Full message history: System, Human, AI, Tool entries.",
    )

    # ── Input ─────────────────────────────────────────────────────────────
    query: str = Field(default="", description="Raw user query string.")

    # ── Router outputs ────────────────────────────────────────────────────
    detected_category: str | None = Field(
        default=None,
        description="LLM-detected policy category for pre-filtering. None = no filter.",
    )
    router_confidence: Literal["high", "low"] = Field(
        default="low",
        description="high = deterministic keyword match; low = LLM classification.",
    )

    # ── Query transformation ──────────────────────────────────────────────
    query_variants: list[str] = Field(
        default_factory=list,
        description="Original query + 3 LLM expansions.",
    )
    hyde_document: str = Field(
        default="",
        description="Hypothetical policy-language answer for HyDE embedding.",
    )

    # ── Retrieval ─────────────────────────────────────────────────────────
    broad_results: list[ChunkResult] = Field(
        default_factory=list,
        description="Deduplicated pool from all query variants (pre-rerank).",
    )

    # ── Reranking ─────────────────────────────────────────────────────────
    reranked_chunks: list[ChunkResult] = Field(
        default_factory=list,
        description="Top-5 chunks after Cross-Encoder reranking.",
    )

    # ── Pipeline config (tunable via A/B test) ────────────────────────────
    num_queries: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of query expansion variants (lecture: 'experiment with numQueries').",
    )
    top_k: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Broad retrieval top-K per variant (lecture: 'experiment with topK').",
    )
    use_reranking: bool = Field(
        default=True,
        description="Enable/disable cross-encoder reranking for A/B testing.",
    )
    use_summarisation: bool = Field(
        default=False,
        description="If True, compress context with cheap model before generation.",
    )
    max_context_tokens: int = Field(
        default=3000,
        ge=500,
        le=8000,
        description="Hard token budget for context passed to answer LLM.",
    )

    # ── TF-IDF calibration ────────────────────────────────────────────────
    tfidf_top_indices: list[int] = Field(
        default_factory=list,
        description="Chunk indices ranked by TF-IDF (sparse) score, before RRF fusion.",
    )
    rrf_applied: bool = Field(
        default=False,
        description="True once Reciprocal Rank Fusion has merged dense + TF-IDF pools.",
    )
    tfidf_calibration_alpha: float = Field(
        default=0.15,
        description="Blend weight for TF-IDF bonus applied after Cross-Encoder reranking.",
        ge=0.0,
        le=1.0,
    )

    # ── Answer ────────────────────────────────────────────────────────────
    final_answer: str = Field(
        default="",
        description="Grounded LLM answer generated from reranked_chunks.",
    )

    # ── Pipeline metadata ─────────────────────────────────────────────────
    pipeline_error: str = Field(
        default="",
        description="Non-empty if any node encountered a recoverable error.",
    )

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("detected_category")
    @classmethod
    def validate_detected_category(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v if v in VALID_CATEGORIES else None

    @field_validator("query")
    @classmethod
    def query_not_empty_after_set(cls, v: str) -> str:
        # Allow empty default; only reject explicit whitespace-only updates
        if v and not v.strip():
            raise ValueError("query must not be blank whitespace")
        return v.strip()

    @model_validator(mode="after")
    def reranked_subset_of_broad(self) -> "AegisState":
        """Top-5 reranked must not exceed the broad pool size."""
        if len(self.reranked_chunks) > max(len(self.broad_results), 5):
            raise ValueError(
                f"reranked_chunks ({len(self.reranked_chunks)}) "
                f"exceeds broad_results ({len(self.broad_results)})"
            )
        return self
