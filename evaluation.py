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
    "SAMPLE_TEST_QUERIES",
    "precision_at_k_from_results",
    "retrieval_accuracy_summary",
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

    question:           str   = Field(...)
    gold_answer:        str   = Field(...)
    predicted_answer:   str   = Field(default="")
    exact_match:        float = Field(default=0.0, ge=0.0, le=1.0)
    token_f1:           float = Field(default=0.0, ge=0.0, le=1.0)
    bleu1:              float = Field(default=0.0, ge=0.0, le=1.0)
    context_precision:  float = Field(default=0.0, ge=0.0, le=1.0)
    context_recall:     float = Field(default=0.0, ge=0.0, le=1.0)
    precision_at_1:     float = Field(default=0.0, ge=0.0, le=1.0,
                                      description="P@1: top source contains gold token")
    precision_at_3:     float = Field(default=0.0, ge=0.0, le=1.0,
                                      description="P@3: fraction of top-3 containing gold")
    precision_at_5:     float = Field(default=0.0, ge=0.0, le=1.0,
                                      description="P@5: fraction of top-5 containing gold")
    retrieval_accuracy: float = Field(default=0.0, ge=0.0, le=1.0,
                                      description="Hits@K: gold doc present in top-K")
    answer_found:       bool  = Field(default=False)
    is_fallback:        bool  = Field(default=False,
                                      description="True if pipeline triggered fallback")
    hallucination_risk: bool  = Field(default=False,
                                      description="True if hallucination check flagged answer")
    latency_s:          float = Field(default=0.0, ge=0.0)
    retrieved_count:    int   = Field(default=0, ge=0)
    category_correct:   bool  = Field(default=False)


class EvalReport(BaseModel):
    """Aggregated evaluation metrics across all truth entries."""

    config_name:           str             = Field(default="default")
    total_questions:       int             = Field(default=0)
    results:               list[EvalResult]= Field(default_factory=list)

    avg_exact_match:       float = Field(default=0.0)
    avg_token_f1:          float = Field(default=0.0)
    avg_bleu1:             float = Field(default=0.0)
    avg_context_precision: float = Field(default=0.0)
    avg_context_recall:    float = Field(default=0.0)
    avg_precision_at_1:    float = Field(default=0.0, description="Mean P@1")
    avg_precision_at_3:    float = Field(default=0.0, description="Mean P@3")
    avg_precision_at_5:    float = Field(default=0.0, description="Mean P@5")
    avg_retrieval_accuracy:float = Field(default=0.0, description="Mean Hits@K")
    avg_latency_s:         float = Field(default=0.0)
    answer_found_rate:     float = Field(default=0.0)
    category_accuracy:     float = Field(default=0.0)
    fallback_rate:         float = Field(default=0.0,
                                        description="Fraction of queries that triggered fallback")
    hallucination_rate:    float = Field(default=0.0,
                                        description="Fraction of answers flagged as hallucination risk")

    def summary(self) -> str:
        return (
            f"[{self.config_name}]\n"
            f"  EM={self.avg_exact_match:.3f}  "
            f"F1={self.avg_token_f1:.3f}  "
            f"BLEU1={self.avg_bleu1:.3f}\n"
            f"  CtxPrec={self.avg_context_precision:.3f}  "
            f"CtxRecall={self.avg_context_recall:.3f}\n"
            f"  P@1={self.avg_precision_at_1:.3f}  "
            f"P@3={self.avg_precision_at_3:.3f}  "
            f"P@5={self.avg_precision_at_5:.3f}\n"
            f"  RetAcc={self.avg_retrieval_accuracy:.3f}  "
            f"AnswerFound={self.answer_found_rate:.3f}\n"
            f"  CatAcc={self.category_accuracy:.3f}  "
            f"Fallback={self.fallback_rate:.3f}  "
            f"HallucRisk={self.hallucination_rate:.3f}  "
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


def _precision_at_k(sources: list[dict], gold_answer: str, k: int) -> float:
    """
    Precision@K: fraction of top-K sources that contain at least one
    gold answer token.

    A source is considered relevant if its chunk_text contains any token
    from the gold answer that is >3 characters (excludes stopwords and
    trivial words).

    Args:
        sources:     Retrieved + reranked source dicts (best first).
        gold_answer: Ground-truth answer string.
        k:           Cutoff rank.

    Returns:
        float in [0, 1].
    """
    if not sources or not gold_answer:
        return 0.0
    gold_tokens = {
        t.lower().strip(".,?!()")
        for t in gold_answer.split()
        if len(t) > 3
    }
    if not gold_tokens:
        return 0.0
    top_k = sources[:k]
    hits  = sum(
        1 for s in top_k
        if any(tok in s.get("chunk_text", "").lower() for tok in gold_tokens)
    )
    return hits / max(len(top_k), 1)


def _retrieval_accuracy(sources: list[dict], gold_doc_ids: list[str],
                        gold_answer: str, k: int = 5) -> float:
    """
    Retrieval Accuracy (Hits@K): 1.0 if either:
      a) Any gold document ID appears in the top-K sources, OR
      b) Any top-K source contains a gold answer token.

    Returns 0.0 if neither condition is met.
    This measures whether the retrieval step surfaced the right content,
    regardless of the final generated answer.
    """
    if not sources:
        return 0.0
    top_k = sources[:k]
    # Condition a: gold doc ID present
    if gold_doc_ids:
        retrieved_ids = {s.get("document_id", "") for s in top_k}
        if any(gid in retrieved_ids for gid in gold_doc_ids):
            return 1.0
    # Condition b: gold answer token present in chunk text
    gold_tokens = {
        t.lower().strip(".,?!()")
        for t in gold_answer.split()
        if len(t) > 3
    }
    if gold_tokens and any(
        any(tok in s.get("chunk_text", "").lower() for tok in gold_tokens)
        for s in top_k
    ):
        return 1.0
    return 0.0


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

        is_fallback    = out.get("is_fallback", False)
        halluc_risk    = out.get("hallucination_risk", False)

        result = EvalResult(
            question=entry.question,
            gold_answer=entry.gold_answer,
            predicted_answer=pred,
            exact_match=_exact_match(pred, entry.gold_answer),
            token_f1=_token_f1(pred, entry.gold_answer),
            bleu1=_bleu1(pred, entry.gold_answer),
            context_precision=_context_precision(sources, entry.gold_answer),
            context_recall=_context_recall(sources, entry.gold_doc_ids),
            precision_at_1=_precision_at_k(sources, entry.gold_answer, k=1),
            precision_at_3=_precision_at_k(sources, entry.gold_answer, k=3),
            precision_at_5=_precision_at_k(sources, entry.gold_answer, k=5),
            retrieval_accuracy=_retrieval_accuracy(
                sources, entry.gold_doc_ids, entry.gold_answer, k=5
            ),
            answer_found=_answer_found(pred),
            is_fallback=is_fallback,
            hallucination_risk=halluc_risk,
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
        avg_exact_match=        sum(r.exact_match          for r in results) / n,
        avg_token_f1=           sum(r.token_f1             for r in results) / n,
        avg_bleu1=              sum(r.bleu1                for r in results) / n,
        avg_context_precision=  sum(r.context_precision    for r in results) / n,
        avg_context_recall=     sum(r.context_recall       for r in results) / n,
        avg_precision_at_1=     sum(r.precision_at_1       for r in results) / n,
        avg_precision_at_3=     sum(r.precision_at_3       for r in results) / n,
        avg_precision_at_5=     sum(r.precision_at_5       for r in results) / n,
        avg_retrieval_accuracy= sum(r.retrieval_accuracy   for r in results) / n,
        avg_latency_s=          sum(r.latency_s            for r in results) / n,
        answer_found_rate=      sum(r.answer_found         for r in results) / n,
        category_accuracy=      sum(r.category_correct     for r in results) / n,
        fallback_rate=          sum(r.is_fallback          for r in results) / n,
        hallucination_rate=     sum(r.hallucination_risk   for r in results) / n,
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


# ---------------------------------------------------------------------------
# SAMPLE TEST QUERIES
# A representative set of 20 queries covering all policy categories.
# Use with run_eval() or ab_test() to benchmark pipeline configurations.
# ---------------------------------------------------------------------------

SAMPLE_TEST_QUERIES: list[str] = [
    # ── Travel ────────────────────────────────────────────────────────────
    "What is the maximum hotel cost per night for domestic travel?",
    "How many days do I have to submit an expense report after travel?",
    "Can I book a business class flight for international trips?",
    "What is the per diem rate for international travel?",
    "Do I need approval before booking a trip that costs $5,000?",
    "Can I expense a taxi to the airport?",
    "What portal must I use to book corporate travel?",
    "What is the minimum passport validity required for international travel?",
    "Is Uber reimbursable as ground transportation?",
    "What happens if I book travel through Expedia instead of the corporate portal?",
    # ── HR ────────────────────────────────────────────────────────────────
    "How long is maternity leave and is it paid?",
    "How many days in advance must I submit a leave request?",
    "What is the policy on sick leave?",
    "Can I carry over unused vacation days to the next year?",
    "What is the tuition reimbursement limit per year?",
    # ── IT / Security ─────────────────────────────────────────────────────
    "What are the password requirements for corporate accounts?",
    "Is personal device use allowed for company work?",
    "What VPN must I use when working remotely?",
    # ── Finance ───────────────────────────────────────────────────────────
    "What is the process for submitting an invoice over $10,000?",
    # ── Cross-domain (ambiguous — tests router) ───────────────────────────
    "What are my rights as an employee regarding data privacy?",
]


# ---------------------------------------------------------------------------
# Summary helpers for Precision@K analysis
# ---------------------------------------------------------------------------

def precision_at_k_from_results(results: list[EvalResult]) -> dict:
    """
    Compute mean Precision@1/3/5 from a list of EvalResult objects.
    Convenience wrapper for reporting.
    """
    n = len(results) or 1
    return {
        "mean_P@1": round(sum(r.precision_at_1 for r in results) / n, 4),
        "mean_P@3": round(sum(r.precision_at_3 for r in results) / n, 4),
        "mean_P@5": round(sum(r.precision_at_5 for r in results) / n, 4),
    }


def retrieval_accuracy_summary(results: list[EvalResult]) -> dict:
    """
    Compute retrieval accuracy stats from a list of EvalResult objects.
    """
    n = len(results) or 1
    acc_values = [r.retrieval_accuracy for r in results]
    return {
        "mean_retrieval_accuracy": round(sum(acc_values) / n, 4),
        "perfect_retrieval_rate": round(sum(1 for v in acc_values if v == 1.0) / n, 4),
        "zero_retrieval_rate":    round(sum(1 for v in acc_values if v == 0.0) / n, 4),
        "n_questions":            n,
    }
