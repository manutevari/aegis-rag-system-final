# =============================================================================
# AEGIS — EVALUATION MODULE
# evaluation.py
#
# Lecture points addressed:
#   "A/B test retrieval strategies (with/without re-ranking, varying query
#    expansion and topK)" (Practical Exercises #1)
#   "RAG evaluation metrics: precision/recall, BLEU, EM-style truth sets"
#    (Evaluation & Upcoming Topics #5)
#   "Maintain a truth set for evaluation" (RAG #7)
#
# What this module provides:
#   1. TruthSet   — Pydantic model for a labelled Q&A pair
#   2. EvalResult — per-question scored result
#   3. EvalReport — aggregated metrics across a truth set
#   4. run_eval() — executes one pipeline configuration against the truth set
#   5. ab_test()  — runs two configs and returns a side-by-side comparison
#
# Metrics computed:
#   • Exact Match (EM)       — normalised string match (lecture: "EM-style")
#   • Token F1               — token-overlap F1 between answer and gold
#   • Context Precision      — fraction of retrieved chunks containing gold
#   • Context Recall         — fraction of gold docs present in retrieved set
#   • BLEU-1                 — unigram precision (lecture: "BLEU")
#   • Answer Found Rate      — did the system return a non-fallback answer?
#
# Pydantic enforcement:
#   Every TruthSet entry and EvalResult is validated on construction.
# =============================================================================

from __future__ import annotations

import re
import string
import time
from collections import Counter
from typing import Callable

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "TruthEntry",
    "EvalResult",
    "EvalReport",
    "run_eval",
    "ab_test",
    "DEFAULT_TRUTH_SET",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TruthEntry(BaseModel):
    """One labelled question-answer pair for evaluation."""

    question:         str       = Field(..., min_length=3)
    gold_answer:      str       = Field(..., min_length=1,
                                        description="Expected answer text or key phrase")
    gold_doc_ids:     list[str] = Field(default_factory=list,
                                        description="Expected source document IDs")
    category:         str       = Field(default="",
                                        description="Expected policy category")

    @field_validator("question", "gold_answer")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


class EvalResult(BaseModel):
    """Scored result for one truth entry."""

    question:          str   = Field(...)
    gold_answer:       str   = Field(...)
    predicted_answer:  str   = Field(default="")
    exact_match:       float = Field(default=0.0, ge=0.0, le=1.0)
    token_f1:          float = Field(default=0.0, ge=0.0, le=1.0)
    bleu1:             float = Field(default=0.0, ge=0.0, le=1.0)
    context_precision: float = Field(default=0.0, ge=0.0, le=1.0)
    context_recall:    float = Field(default=0.0, ge=0.0, le=1.0)
    answer_found:      bool  = Field(default=False)
    latency_s:         float = Field(default=0.0, ge=0.0)
    retrieved_count:   int   = Field(default=0, ge=0)
    category_correct:  bool  = Field(default=False)


class EvalReport(BaseModel):
    """Aggregated evaluation metrics across all truth entries."""

    config_name:       str         = Field(default="default")
    total_questions:   int         = Field(default=0)
    results:           list[EvalResult] = Field(default_factory=list)

    avg_exact_match:       float = Field(default=0.0)
    avg_token_f1:          float = Field(default=0.0)
    avg_bleu1:             float = Field(default=0.0)
    avg_context_precision: float = Field(default=0.0)
    avg_context_recall:    float = Field(default=0.0)
    avg_latency_s:         float = Field(default=0.0)
    answer_found_rate:     float = Field(default=0.0)
    category_accuracy:     float = Field(default=0.0)

    def summary(self) -> str:
        return (
            f"[{self.config_name}] "
            f"EM={self.avg_exact_match:.3f}  "
            f"F1={self.avg_token_f1:.3f}  "
            f"BLEU1={self.avg_bleu1:.3f}  "
            f"CtxPrec={self.avg_context_precision:.3f}  "
            f"CtxRecall={self.avg_context_recall:.3f}  "
            f"AnswerFound={self.answer_found_rate:.3f}  "
            f"CatAcc={self.category_accuracy:.3f}  "
            f"Latency={self.avg_latency_s:.2f}s"
        )


# ---------------------------------------------------------------------------
# Text normalisation helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokens(text: str) -> list[str]:
    return _normalise(text).split()


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalise(pred) == _normalise(gold) else 0.0


def _token_f1(pred: str, gold: str) -> float:
    """Standard SQuAD-style token F1."""
    pred_toks = Counter(_tokens(pred))
    gold_toks = Counter(_tokens(gold))
    common    = sum((pred_toks & gold_toks).values())
    if common == 0:
        return 0.0
    precision = common / sum(pred_toks.values())
    recall    = common / sum(gold_toks.values())
    return 2 * precision * recall / (precision + recall)


def _bleu1(pred: str, gold: str) -> float:
    """Unigram BLEU (clipped precision)."""
    pred_toks  = _tokens(pred)
    gold_toks  = set(_tokens(gold))
    if not pred_toks:
        return 0.0
    matches = sum(1 for t in pred_toks if t in gold_toks)
    return matches / len(pred_toks)


def _context_precision(sources: list[dict], gold_answer: str) -> float:
    """Fraction of retrieved chunks that contain at least one gold token."""
    if not sources:
        return 0.0
    gold_toks = set(_tokens(gold_answer))
    if not gold_toks:
        return 0.0
    hits = sum(
        1 for s in sources
        if gold_toks & set(_tokens(s.get("chunk_text", "")))
    )
    return hits / len(sources)


def _context_recall(sources: list[dict], gold_doc_ids: list[str]) -> float:
    """Fraction of expected document IDs that appear in retrieved sources."""
    if not gold_doc_ids:
        return 1.0   # no constraint → vacuously true
    retrieved_ids = {s.get("document_id", "") for s in sources}
    hits = sum(1 for gid in gold_doc_ids if gid in retrieved_ids)
    return hits / len(gold_doc_ids)


_FALLBACK_PHRASES = {
    "could not find", "not found", "no relevant", "unavailable",
    "i don't know", "unable to answer",
}

def _answer_found(answer: str) -> bool:
    low = answer.lower()
    return not any(phrase in low for phrase in _FALLBACK_PHRASES)


# ---------------------------------------------------------------------------
# Core evaluation runner
# ---------------------------------------------------------------------------

def run_eval(
    truth_set:   list[TruthEntry],
    pipeline_fn: Callable[[str], dict],
    config_name: str = "default",
) -> EvalReport:
    """
    Evaluate a pipeline function against a labelled truth set.

    Args:
        truth_set:   List of TruthEntry objects.
        pipeline_fn: Callable that takes a query string and returns a dict
                     with keys: answer, category, sources (list of chunk dicts).
        config_name: Label for this config in the report.

    Returns:
        EvalReport with per-question results and aggregate metrics.
    """
    results: list[EvalResult] = []

    for entry in truth_set:
        t0 = time.time()
        try:
            out = pipeline_fn(entry.question)
        except Exception as exc:
            out = {"answer": f"ERROR: {exc}", "category": None, "sources": []}
        latency = round(time.time() - t0, 3)

        pred    = out.get("answer", "")
        sources = out.get("sources", [])
        cat     = out.get("category") or ""

        result = EvalResult(
            question=entry.question,
            gold_answer=entry.gold_answer,
            predicted_answer=pred,
            exact_match=_exact_match(pred, entry.gold_answer),
            token_f1=_token_f1(pred, entry.gold_answer),
            bleu1=_bleu1(pred, entry.gold_answer),
            context_precision=_context_precision(sources, entry.gold_answer),
            context_recall=_context_recall(sources, entry.gold_doc_ids),
            answer_found=_answer_found(pred),
            latency_s=latency,
            retrieved_count=len(sources),
            category_correct=(
                cat.lower() == entry.category.lower()
                if entry.category else True
            ),
        )
        results.append(result)

    n = len(results) or 1
    report = EvalReport(
        config_name=config_name,
        total_questions=len(results),
        results=results,
        avg_exact_match=       sum(r.exact_match       for r in results) / n,
        avg_token_f1=          sum(r.token_f1          for r in results) / n,
        avg_bleu1=             sum(r.bleu1             for r in results) / n,
        avg_context_precision= sum(r.context_precision for r in results) / n,
        avg_context_recall=    sum(r.context_recall    for r in results) / n,
        avg_latency_s=         sum(r.latency_s         for r in results) / n,
        answer_found_rate=     sum(r.answer_found      for r in results) / n,
        category_accuracy=     sum(r.category_correct  for r in results) / n,
    )
    return report


# ---------------------------------------------------------------------------
# A/B test harness
# ---------------------------------------------------------------------------

def ab_test(
    truth_set:     list[TruthEntry],
    pipeline_a:    Callable[[str], dict],
    pipeline_b:    Callable[[str], dict],
    name_a:        str = "Config-A",
    name_b:        str = "Config-B",
) -> tuple[EvalReport, EvalReport, dict]:
    """
    Run two pipeline configurations against the same truth set and compare.

    Lecture point: "A/B test retrieval strategies (with/without re-ranking,
    varying query expansion and topK)" (Homework #1)

    Returns:
        (report_a, report_b, comparison_dict)
        comparison_dict contains per-metric delta (B - A) and winner.
    """
    report_a = run_eval(truth_set, pipeline_a, config_name=name_a)
    report_b = run_eval(truth_set, pipeline_b, config_name=name_b)

    metrics = [
        "avg_exact_match", "avg_token_f1", "avg_bleu1",
        "avg_context_precision", "avg_context_recall",
        "answer_found_rate", "category_accuracy",
    ]
    comparison: dict = {"name_a": name_a, "name_b": name_b, "metrics": {}}

    for m in metrics:
        a_val = getattr(report_a, m)
        b_val = getattr(report_b, m)
        delta = round(b_val - a_val, 4)
        comparison["metrics"][m] = {
            name_a: round(a_val, 4),
            name_b: round(b_val, 4),
            "delta_B_minus_A": delta,
            "winner": name_b if delta > 0 else (name_a if delta < 0 else "tie"),
        }

    # Latency: lower is better — flip winner logic
    a_lat = report_a.avg_latency_s
    b_lat = report_b.avg_latency_s
    comparison["metrics"]["avg_latency_s"] = {
        name_a: round(a_lat, 3),
        name_b: round(b_lat, 3),
        "delta_B_minus_A": round(b_lat - a_lat, 3),
        "winner": name_a if b_lat > a_lat else (name_b if b_lat < a_lat else "tie"),
    }

    b_wins = sum(
        1 for v in comparison["metrics"].values() if v["winner"] == name_b
    )
    comparison["overall_winner"] = (
        name_b if b_wins > len(comparison["metrics"]) / 2 else name_a
    )

    return report_a, report_b, comparison


# ---------------------------------------------------------------------------
# Default truth set — 12 questions from the project's policy documents
# ---------------------------------------------------------------------------

DEFAULT_TRUTH_SET: list[TruthEntry] = [
    TruthEntry(
        question="What is the maximum hotel cost per night for domestic travel?",
        gold_answer="$200 per night",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
    TruthEntry(
        question="How many days do employees have to submit expense reports?",
        gold_answer="30 days",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
    TruthEntry(
        question="What approval is needed for a trip costing $5,000?",
        gold_answer="Vice President and Finance Controller",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
    TruthEntry(
        question="What is the economy class threshold for flight duration?",
        gold_answer="6 hours",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
    TruthEntry(
        question="How long is maternity leave?",
        gold_answer="16 weeks",
        gold_doc_ids=[],
        category="HR",
    ),
    TruthEntry(
        question="How many days in advance must leave be requested?",
        gold_answer="14 days",
        gold_doc_ids=[],
        category="HR",
    ),
    TruthEntry(
        question="What is the per diem rate for international travel?",
        gold_answer="$75 per day",
        gold_doc_ids=[],
        category="Travel",
    ),
    TruthEntry(
        question="What portal must be used to book all corporate travel?",
        gold_answer="TripIt Corporate Navigator",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
    TruthEntry(
        question="What is the minimum passport validity required for international travel?",
        gold_answer="six months",
        gold_doc_ids=[],
        category="Travel",
    ),
    TruthEntry(
        question="Who is responsible for managing travel vendor relationships?",
        gold_answer="GCTEM",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
    TruthEntry(
        question="What is the maximum taxi reimbursement per trip?",
        gold_answer="$50 per trip",
        gold_doc_ids=[],
        category="Travel",
    ),
    TruthEntry(
        question="What is a Level 1 Policy Violation for travel booking?",
        gold_answer="Booking through external third-party consumer sites",
        gold_doc_ids=["TRV-POL-1001-V4"],
        category="Travel",
    ),
]
