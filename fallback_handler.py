# =============================================================================
# AEGIS — FALLBACK HANDLER
# fallback_handler.py
#
# Centralises ALL fallback logic for the pipeline.
# Previously fallbacks were scattered across nodes with bare strings.
# Now every fallback is typed, logged, and traced.
#
# Three categories of fallback:
#   1. EMPTY_RESULTS  — retrieval returned nothing
#   2. HALLUCINATION_RISK — answer contains terms absent from context
#   3. API_ERROR      — upstream LLM / Pinecone call failed
#
# Anti-hallucination check:
#   After generation, scan the answer for noun phrases that don't appear
#   in ANY retrieved chunk. If >HALLUC_THRESHOLD fraction of answer tokens
#   are "foreign" (not in context vocabulary), flag as hallucination risk
#   and replace with the safe fallback answer.
#
# Pydantic enforcement:
#   FallbackResponse wraps every fallback so consumers always get a
#   validated dict with a known schema.
# =============================================================================

from __future__ import annotations

import re
import string
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from graph_state import ToolMessage, tool_log
from logger import get_logger

log = get_logger(__name__)

__all__ = [
    "FallbackReason",
    "FallbackResponse",
    "handle_fallback",
    "check_hallucination_risk",
    "SAFE_FALLBACK_ANSWER",
    "NOT_FOUND_PHRASE",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The EXACT phrase the system prompt requires when context is insufficient.
# Used by node_generate's system prompt AND by check_hallucination_risk.
NOT_FOUND_PHRASE = "Not found in context"

# Full safe answer returned when fallback triggers
SAFE_FALLBACK_ANSWER = (
    "Not found in context. "
    "I could not locate relevant policy information to answer your question. "
    "Please ensure the relevant document has been ingested, or try rephrasing "
    "your question with more specific policy terms."
)

# Fraction of answer tokens not present in context vocabulary that triggers
# hallucination flag (conservative: 0.35 = 35% foreign tokens)
HALLUC_THRESHOLD = 0.35


# ---------------------------------------------------------------------------
# FallbackReason enum
# ---------------------------------------------------------------------------

class FallbackReason(str, Enum):
    EMPTY_QUERY     = "empty_query"
    EMPTY_RESULTS   = "empty_results"
    HALLUCINATION   = "hallucination_risk"
    API_ERROR       = "api_error"
    TIMEOUT         = "timeout"
    CONTEXT_MISSING = "context_missing"
    IMPORT_ERROR    = "import_error"
    UNKNOWN         = "unknown"


# ---------------------------------------------------------------------------
# FallbackResponse Pydantic model
# ---------------------------------------------------------------------------

class FallbackResponse(BaseModel):
    """
    Validated wrapper for any fallback event.
    Consumers can always rely on 'answer' and 'reason' being present.
    """
    answer:         str          = Field(default=SAFE_FALLBACK_ANSWER)
    reason:         FallbackReason = Field(default=FallbackReason.UNKNOWN)
    original_query: str          = Field(default="")
    node:           str          = Field(default="unknown")
    detail:         str          = Field(default="")
    halluc_score:   float        = Field(default=0.0, ge=0.0, le=1.0)
    is_fallback:    bool         = Field(default=True)

    def to_tool_message(self) -> ToolMessage:
        return tool_log(
            tool_name="fallback_handler",
            reason=f"Fallback triggered in '{self.node}': {self.reason.value}. {self.detail}",
            inputs={"query": self.original_query, "node": self.node},
            outputs={
                "reason":      self.reason.value,
                "halluc_score": self.halluc_score,
                "answer":      self.answer[:200],
            },
        )


# ---------------------------------------------------------------------------
# Main fallback factory
# ---------------------------------------------------------------------------

def handle_fallback(
    reason:  FallbackReason,
    node:    str,
    query:   str = "",
    detail:  str = "",
    answer:  str | None = None,
    halluc_score: float = 0.0,
) -> FallbackResponse:
    """
    Create a FallbackResponse, log it, and return it.
    Call this whenever a node must return a degraded answer.

    Args:
        reason:       Why the fallback triggered (FallbackReason enum).
        node:         Name of the pipeline node where fallback occurred.
        query:        The user query (for logging).
        detail:       Human-readable explanation of the specific failure.
        answer:       Override answer text. Defaults to SAFE_FALLBACK_ANSWER.
        halluc_score: Hallucination risk score [0,1], if computed.

    Returns:
        FallbackResponse with is_fallback=True.
    """
    resp = FallbackResponse(
        answer=answer or SAFE_FALLBACK_ANSWER,
        reason=reason,
        original_query=query[:80],
        node=node,
        detail=detail,
        halluc_score=halluc_score,
        is_fallback=True,
    )
    log.warning(
        f"FALLBACK [{reason.value}] in {node}: {detail[:120]}",
        extra={"run_id": "n/a", "query": query[:60], "node": node,
               "event": "fallback", "reason": reason.value},
    )
    return resp


# ---------------------------------------------------------------------------
# Anti-hallucination check
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> set[str]:
    """Lowercase tokens, strip punctuation, filter stopwords."""
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "shall", "can",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "that", "this", "it", "its", "or", "and", "but", "not", "no",
        "as", "if", "then", "than", "so", "yet", "nor", "both", "each",
        "per", "any", "all", "more", "most", "other", "such", "which",
        "when", "where", "who", "how", "what", "why", "i", "you", "we",
        "they", "he", "she", "our", "their", "your", "my", "his", "her",
    }
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return {t for t in text.split() if len(t) > 2 and t not in _STOP}


def check_hallucination_risk(
    answer:  str,
    context_chunks: list[str],
    threshold: float = HALLUC_THRESHOLD,
) -> tuple[bool, float]:
    """
    Detect whether the answer contains terms that don't appear in the
    retrieved context (potential hallucination signal).

    Method:
      1. Build a vocabulary from all retrieved chunk texts.
      2. Tokenise the answer.
      3. Count answer tokens NOT in the context vocabulary.
      4. If the ratio exceeds `threshold`, flag as hallucination risk.

    This is a heuristic, not a definitive test. False positives occur for
    paraphrasing; false negatives for copied verbatim text. It catches the
    most common case: fabricated names, dates, and numbers.

    Args:
        answer:         The generated answer string.
        context_chunks: List of raw chunk texts used as context.
        threshold:      Foreign-token ratio that triggers flag (default 0.35).

    Returns:
        (is_risky: bool, score: float)
        score ∈ [0, 1] — higher = more foreign tokens in answer.
    """
    if not answer or not context_chunks:
        return False, 0.0

    # Skip if answer explicitly says "Not found"
    if NOT_FOUND_PHRASE.lower() in answer.lower():
        return False, 0.0

    context_vocab = set()
    for chunk in context_chunks:
        context_vocab |= _tokenise(chunk)

    answer_tokens = _tokenise(answer)
    if not answer_tokens:
        return False, 0.0

    foreign = answer_tokens - context_vocab
    score   = len(foreign) / len(answer_tokens)
    is_risky = score > threshold

    if is_risky:
        log.warning(
            f"Hallucination risk: {score:.2%} of answer tokens not in context "
            f"(threshold={threshold:.0%}). Foreign: {sorted(foreign)[:8]}",
            extra={"event": "hallucination_check", "score": score,
                   "threshold": threshold, "foreign_sample": list(foreign)[:8]},
        )

    return is_risky, round(score, 4)
