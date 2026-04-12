# =============================================================================
# AEGIS — TF-IDF SPARSE INDEX
# tfidf_index.py
#
# Purpose: calibrate retrieval precision by adding a sparse TF-IDF scoring
# layer that rewards exact-term overlap between query and chunk.
#
# Why TF-IDF improves precision here:
#   Dense (semantic) embeddings have excellent recall — they find paraphrases
#   and conceptually related chunks — but poor precision for exact policy terms:
#   dollar amounts ("$1,000"), day counts ("30 days"), acronyms ("GCTEM",
#   "MSA"), and section names ("4.2 Booking Channels") that don't cluster well
#   in embedding space.
#
#   TF-IDF assigns high weight to rare, informative terms (IDF penalises
#   common words) and rewards chunks where those terms appear frequently (TF).
#   The result: queries with specific numbers or jargon get precision-boosted
#   candidates even when semantic distance is moderate.
#
# Integration points:
#   1. node_retrieve  — TfidfRetriever runs in parallel with dense retrieval;
#                       both pools are merged via Reciprocal Rank Fusion (RRF)
#                       before deduplication. RRF is parameter-free and robust.
#
#   2. node_rerank    — after Cross-Encoder scoring, a TF-IDF term-overlap
#                       calibration bonus is applied. Chunks that contain the
#                       exact query terms get a small upward nudge that can
#                       break ties and surface numerically precise chunks.
#
# Design decisions:
#   • Fits TF-IDF on the dense candidate pool (not the whole corpus).
#     No offline index needed; works on any set of ChunkResult objects.
#   • Uses sklearn TfidfVectorizer with sublinear_tf=True (log-TF) and
#     min_df=1. Policy corpora are small enough that this is fine.
#   • Scores are normalised to [0, 1] before fusion so TF-IDF and dense
#     scores contribute equally to the final ranking.
#   • All decisions logged as ToolMessage via tool_log().
#
# Pydantic enforcement:
#   TfidfResult wraps each scored chunk; invalid float scores are coerced
#   to 0.0 rather than raising so a bad TF-IDF score never kills the run.
# =============================================================================

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from graph_state import ChunkResult

__all__ = [
    "TfidfIndex",
    "TfidfResult",
    "reciprocal_rank_fusion",
    "tfidf_calibrate_scores",
]


# ---------------------------------------------------------------------------
# Pydantic model for a TF-IDF scored chunk
# ---------------------------------------------------------------------------

class TfidfResult(BaseModel):
    """A chunk with its TF-IDF cosine similarity score attached."""

    chunk_index: int   = Field(..., description="Index into the source chunk list")
    tfidf_score: float = Field(default=0.0, description="Cosine similarity [0, 1]")

    @field_validator("tfidf_score", mode="before")
    @classmethod
    def coerce_score(cls, v) -> float:
        """Coerce any non-finite or out-of-range score to 0.0 safely."""
        try:
            f = float(v)
            return f if math.isfinite(f) and f >= 0.0 else 0.0
        except (TypeError, ValueError):
            return 0.0


# ---------------------------------------------------------------------------
# TF-IDF Index
# ---------------------------------------------------------------------------

class TfidfIndex:
    """
    Lightweight TF-IDF index that fits on an arbitrary list of text chunks.

    Typical usage (in node_retrieve):
        idx = TfidfIndex(corpus_texts)
        ranked = idx.query(query_text, top_k=25)

    Typical usage (in node_rerank for calibration):
        idx   = TfidfIndex([c.chunk_text for c in broad_results])
        bonus = idx.score_all(query_text)   # array of length len(broad_results)
    """

    def __init__(self, texts: list[str]) -> None:
        """
        Fit a TfidfVectorizer on `texts`.

        Args:
            texts: List of document strings to index.
                   May be empty — index becomes a no-op scorer.
        """
        self._texts  = texts
        self._matrix = None   # sparse (n_docs × n_features) matrix
        self._vec    = None   # fitted TfidfVectorizer

        if not texts:
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(
                sublinear_tf=True,      # log(1+tf) dampens high-freq terms
                min_df=1,               # keep even hapax legomena (policy jargon)
                max_df=0.95,            # drop tokens present in >95% of docs
                ngram_range=(1, 2),     # unigrams + bigrams for phrase matching
                                        # ("per diem", "booking channel", etc.)
                strip_accents="unicode",
                analyzer="word",
                token_pattern=r"(?u)\b[\w\$\%\.\/\-]{2,}\b",
                                        # keep $, %, ., / in tokens so
                                        # "$1,000" and "4.2" are preserved
            )
            self._matrix = self._vec.fit_transform(texts)
        except ImportError:
            # sklearn not installed — index silently becomes a no-op
            pass

    # ── Public API ────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 25) -> list[TfidfResult]:
        """
        Return up to top_k TfidfResult objects sorted by descending score.
        Returns empty list if the index is empty or sklearn is unavailable.
        """
        scores = self.score_all(query_text)
        if not scores:
            return []

        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [
            TfidfResult(chunk_index=i, tfidf_score=s)
            for i, s in indexed[:top_k]
            if s > 0.0          # skip zero-similarity results
        ]

    def score_all(self, query_text: str) -> list[float]:
        """
        Return a float score for every document in the corpus.
        Scores are cosine similarities in [0, 1].
        Returns a list of zeros if the index is empty.
        """
        if self._vec is None or self._matrix is None:
            return [0.0] * len(self._texts)

        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity

            q_vec   = self._vec.transform([query_text])   # (1 × n_features)
            sims    = cosine_similarity(q_vec, self._matrix).flatten()  # (n_docs,)
            return [float(s) for s in sims]
        except Exception:
            return [0.0] * len(self._texts)

    @property
    def is_ready(self) -> bool:
        return self._vec is not None and self._matrix is not None

    def __len__(self) -> int:
        return len(self._texts)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF)
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    dense_chunks:  list["ChunkResult"],
    tfidf_results: list[TfidfResult],
    k: int = 60,
) -> list["ChunkResult"]:
    """
    Merge dense retrieval results and TF-IDF results using Reciprocal Rank
    Fusion (RRF).

    RRF score for document d:  Σ  1 / (k + rank_i(d))
    where rank_i(d) is 1-based rank in list i, and k=60 is the standard
    parameter (Cormack et al. 2009) that dampens the influence of very
    high-ranked documents.

    Why RRF over score normalisation:
      • Dense scores (cosine) and TF-IDF scores (cosine) have different
        distributions. Normalising by max/min is fragile when one list has
        extreme outliers. RRF depends only on rank, not magnitude.
      • Parameter-free except for k; k=60 works well across retrieval tasks.

    Args:
        dense_chunks:  ChunkResult list ordered by dense retrieval (best first).
        tfidf_results: TfidfResult list ordered by TF-IDF score (best first).
                       chunk_index references into dense_chunks.
        k:             RRF smoothing constant (default 60).

    Returns:
        Re-ordered list of ChunkResult (RRF-best first), with
        vector_score updated to the RRF fusion score for auditability.
    """
    from graph_state import ChunkResult as CR   # local import to avoid circular

    rrf: dict[int, float] = {}

    # Dense ranking contribution
    for rank, chunk in enumerate(dense_chunks, start=1):
        rrf[id(chunk)] = rrf.get(id(chunk), 0.0) + 1.0 / (k + rank)

    # TF-IDF ranking contribution (maps back to dense_chunks by chunk_index)
    for trank, tr in enumerate(tfidf_results, start=1):
        idx = tr.chunk_index
        if 0 <= idx < len(dense_chunks):
            obj_id = id(dense_chunks[idx])
            rrf[obj_id] = rrf.get(obj_id, 0.0) + 1.0 / (k + trank)

    # Sort by fused score
    scored = sorted(dense_chunks, key=lambda c: rrf.get(id(c), 0.0), reverse=True)

    # Write RRF score back into vector_score field for audit visibility
    result: list = []
    for c in scored:
        data = c.model_dump()
        data["vector_score"] = round(rrf.get(id(c), 0.0), 6)
        try:
            result.append(CR(**data))
        except Exception:
            result.append(c)   # keep original if re-validation fails

    return result


# ---------------------------------------------------------------------------
# TF-IDF Calibration Bonus (applied after Cross-Encoder scoring)
# ---------------------------------------------------------------------------

def tfidf_calibrate_scores(
    query: str,
    chunks: list["ChunkResult"],
    alpha: float = 0.15,
) -> list["ChunkResult"]:
    """
    Post-rerank TF-IDF calibration: add a weighted TF-IDF term-overlap bonus
    to the Cross-Encoder rerank_score, then re-sort.

    Formula:
        calibrated_score = rerank_score + alpha × tfidf_score

    Why alpha=0.15:
        The cross-encoder score dominates (weight 1.0).
        TF-IDF acts as a tiebreaker / nudge (weight 0.15).
        At alpha=0.15 a perfect TF-IDF match (+0.15) can overcome a
        ~0.15-point cross-encoder gap — enough to surface numerically
        precise chunks without overriding strong semantic matches.

    This is especially important for policy queries that include:
      • Dollar amounts:   "What is the $500 per diem limit?"
      • Day counts:       "Is the 30-day submission window correct?"
      • Section refs:     "What does section 4.2 say?"
      • Acronyms:         "GCTEM approval required?"

    Args:
        query:  Raw user query string.
        chunks: Reranked ChunkResult list (best first by rerank_score).
        alpha:  TF-IDF blend weight. Must be in [0, 1]. Default 0.15.

    Returns:
        New list of ChunkResult sorted by calibrated score (descending),
        with rerank_score updated to the calibrated value for audit.

    Pydantic enforcement:
        Each ChunkResult is re-constructed after score update.
        Invalid calibrated scores fall back to the original rerank_score.
    """
    from graph_state import ChunkResult as CR

    if not chunks:
        return chunks

    # Clamp alpha
    alpha = max(0.0, min(1.0, alpha))

    texts = [c.chunk_text for c in chunks]
    idx   = TfidfIndex(texts)

    if not idx.is_ready:
        # sklearn unavailable — return unchanged
        return chunks

    tf_scores = idx.score_all(query)   # list[float], length = len(chunks)

    calibrated: list = []
    for chunk, tf_s in zip(chunks, tf_scores):
        cal_score = chunk.rerank_score + alpha * tf_s
        data = chunk.model_dump()
        data["rerank_score"] = round(cal_score, 6)
        try:
            calibrated.append(CR(**data))
        except Exception:
            calibrated.append(chunk)

    # Re-sort by calibrated score
    calibrated.sort(key=lambda c: c.rerank_score, reverse=True)
    return calibrated
