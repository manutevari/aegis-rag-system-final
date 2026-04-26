"""
Production-Grade Evaluation Metrics for AEGIS RAG
- Deterministic (fast)
- Safe for all doc types
- ROI-aware (lightweight + reliable)
"""

import re
from difflib import SequenceMatcher
from typing import List, Any


# -----------------------------
# Text Normalization
# -----------------------------
def normalize(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -----------------------------
# Exact Match (Strict)
# -----------------------------
def exact_match(pred: str, truth: str) -> float:
    pred_n = normalize(pred)
    truth_n = normalize(truth)
    return 1.0 if pred_n == truth_n else 0.0


# -----------------------------
# Fuzzy Match (Semantic Approx)
# -----------------------------
def fuzzy_match(pred: str, truth: str) -> float:
    return SequenceMatcher(None, normalize(pred), normalize(truth)).ratio()


# -----------------------------
# Hybrid Score (Recommended)
# -----------------------------
def answer_score(pred: str, truth: str, threshold: float = 0.85) -> float:
    """
    Returns:
    - 1.0 → correct
    - 0.0 → incorrect
    """
    if exact_match(pred, truth) == 1.0:
        return 1.0

    score = fuzzy_match(pred, truth)
    return 1.0 if score >= threshold else 0.0


# -----------------------------
# Recall@K (Retrieval Quality)
# -----------------------------
def recall_at_k(retrieved_docs: List[Any], truth_source: str, k: int = 5) -> float:
    """
    Checks if correct source appears in top-k retrieved docs
    Safe for:
    - dict
    - string
    - LangChain Document
    """
    if not retrieved_docs:
        return 0.0

    top_k = retrieved_docs[:k]

    for doc in top_k:
        try:
            content = str(doc)
            if truth_source.lower() in content.lower():
                return 1.0
        except Exception:
            continue

    return 0.0
