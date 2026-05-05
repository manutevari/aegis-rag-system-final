from __future__ import annotations

import re
from typing import Callable, Dict, Iterable, List, Tuple

STRICT_NOT_FOUND = "This information is not available in the policy."

PROVIDER_PROFILE = {
    "openai": {"cost": 0.55, "adequacy": 0.90, "reasoning": 0.90},
    "grok": {"cost": 0.68, "adequacy": 0.88, "reasoning": 0.92},
    "gemini": {"cost": 0.42, "adequacy": 0.84, "reasoning": 0.86},
    "mistral": {"cost": 0.35, "adequacy": 0.78, "reasoning": 0.80},
    "openrouter": {"cost": 0.48, "adequacy": 0.80, "reasoning": 0.82},
    "huggingface": {"cost": 0.25, "adequacy": 0.72, "reasoning": 0.72},
}
DEFAULT_CONTROLS = {
    "mode": "Balanced",
    "adequacy_threshold": 0.52,
    "max_cloud_attempts": 3,
    "human_review": True,
}
REVIEW_TERMS = {
    "audit",
    "benefit",
    "bonus",
    "compliance",
    "disciplinary",
    "harassment",
    "legal",
    "payroll",
    "privacy",
    "reimbursement",
    "security",
    "termination",
    "visa",
}
STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "because",
    "before",
    "could",
    "from",
    "have",
    "into",
    "only",
    "policy",
    "question",
    "should",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}


def normalized_controls(raw: Dict[str, object] | None = None) -> Dict[str, object]:
    controls = dict(DEFAULT_CONTROLS)
    controls.update(raw or {})
    controls["adequacy_threshold"] = float(controls.get("adequacy_threshold") or 0.52)
    controls["max_cloud_attempts"] = int(controls.get("max_cloud_attempts") or 3)
    controls["human_review"] = bool(controls.get("human_review", True))
    return controls


def routing_weights(mode: str) -> Tuple[float, float]:
    if mode == "Cost efficient":
        return 0.65, 0.35
    if mode == "Highest adequacy":
        return 0.20, 0.80
    return 0.40, 0.60


def default_candidates(sentiment: Dict[str, object], compute_state: Dict[str, object]) -> List[str]:
    candidates = {
        "negative": ["grok", "openai", "gemini", "huggingface", "openrouter", "mistral"],
        "positive": ["gemini", "openai", "huggingface", "grok", "openrouter", "mistral"],
        "neutral": ["openai", "huggingface", "grok", "gemini", "mistral", "openrouter"],
    }.get(
        str(sentiment.get("label")),
        ["openai", "huggingface", "grok", "gemini", "mistral", "openrouter"],
    )
    if compute_state.get("compute_result") is not None:
        candidates = ["openai", "grok", "gemini", "mistral", "openrouter", "huggingface"]
    return candidates


def provider_score(
    provider: str,
    sentiment: Dict[str, object],
    compute_state: Dict[str, object],
    controls: Dict[str, object] | None = None,
) -> float:
    active = normalized_controls(controls)
    profile = PROVIDER_PROFILE.get(provider, {"cost": 0.50, "adequacy": 0.75, "reasoning": 0.75})
    cost_weight, adequacy_weight = routing_weights(str(active["mode"]))
    score = (profile["adequacy"] * adequacy_weight) - (profile["cost"] * cost_weight)

    label = str(sentiment.get("label"))
    if label == "negative" and provider in {"grok", "openai"}:
        score += 0.10
    elif label == "positive" and provider in {"gemini", "openai"}:
        score += 0.07
    elif label == "neutral" and provider in {"openai", "huggingface", "gemini"}:
        score += 0.04

    if compute_state.get("compute_result") is not None:
        score += profile["reasoning"] * 0.14
        if provider in {"openai", "grok", "gemini"}:
            score += 0.08

    return round(score, 4)


def rank_providers(
    sentiment: Dict[str, object],
    compute_state: Dict[str, object],
    controls: Dict[str, object] | None = None,
    candidates: Iterable[str] | None = None,
) -> List[str]:
    ordered = list(candidates or default_candidates(sentiment, compute_state))
    return sorted(
        ordered,
        key=lambda provider: provider_score(provider, sentiment, compute_state, controls),
        reverse=True,
    )


def candidate_chain(
    selected_provider: str,
    sentiment: Dict[str, object],
    compute_state: Dict[str, object],
    cloud_providers: Iterable[str],
    provider_key: Callable[[str], str],
    controls: Dict[str, object] | None = None,
) -> List[str]:
    active = normalized_controls(controls)
    cloud = set(cloud_providers)
    if selected_provider == "auto_sentiment":
        candidates = rank_providers(sentiment, compute_state, active)
    elif selected_provider in cloud:
        fallback_order = rank_providers(sentiment, compute_state, active)
        candidates = [selected_provider] + [provider for provider in fallback_order if provider != selected_provider]
    else:
        return [selected_provider]

    available = [provider for provider in candidates if provider_key(provider)]
    limit = max(1, min(int(active["max_cloud_attempts"]), len(available)))
    chain = available[:limit] or ["extractive"]
    if chain != ["extractive"]:
        chain.append(
            f"orchestration:{active['mode']}:{active['adequacy_threshold']}:"
            f"{active['max_cloud_attempts']}:{active['human_review']}"
        )
    return chain


def content_tokens(text: str) -> set:
    return {
        word
        for word in re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{3,}", text.lower())
        if word not in STOPWORDS
    }


def answer_adequacy_score(answer: str, query: str, context: str, draft: str) -> Tuple[float, str]:
    cleaned = (answer or "").strip()
    if len(cleaned) < 8:
        return 0.0, "empty or too short"

    lower_answer = cleaned.lower()
    lower_draft = (draft or "").lower()
    strict_lower = STRICT_NOT_FOUND.lower()
    failure_phrases = [
        "i do not have access",
        "i don't have access",
        "no context provided",
        "outside the provided context",
    ]
    if any(phrase in lower_answer for phrase in failure_phrases):
        return 0.10, "model ignored or missed supplied context"

    if strict_lower in lower_answer:
        if strict_lower in lower_draft or not (context or "").strip():
            return 0.90, "not-found answer matches retrieved evidence"
        return 0.20, "model returned not-found despite retrieved evidence"

    if strict_lower in lower_draft and not (context or "").strip():
        return 0.15, "draft and context did not support an answer"

    answer_tokens = content_tokens(cleaned)
    context_tokens = content_tokens(context)
    query_tokens = content_tokens(query)
    shared_context = len(answer_tokens & context_tokens)
    shared_query = len(answer_tokens & query_tokens)
    context_overlap = shared_context / max(1, min(len(answer_tokens), len(context_tokens or answer_tokens)))
    query_overlap = shared_query / max(1, len(query_tokens))
    score = 0.30 + min(context_overlap, 1.0) * 0.50 + min(query_overlap, 1.0) * 0.20

    if context_tokens and shared_context < 2 and shared_query < 1:
        score = min(score, 0.35)
        return round(score, 2), "low overlap with retrieved policy context"

    return round(min(score, 1.0), 2), "accepted"


def answer_is_relevant(
    answer: str,
    query: str,
    context: str,
    draft: str,
    controls: Dict[str, object] | None = None,
) -> Tuple[bool, str]:
    active = normalized_controls(controls)
    score, reason = answer_adequacy_score(answer, query, context, draft)
    threshold = float(active["adequacy_threshold"])
    return (
        score >= threshold,
        f"{reason}; adequacy={score:.2f}; threshold={threshold:.2f}; routing={active['mode']}",
    )


def human_review_reasons(
    query: str,
    sentiment: Dict[str, object],
    compute_state: Dict[str, object],
    answer: str,
    context: str,
    draft: str,
    controls: Dict[str, object] | None = None,
) -> Tuple[float, List[str]]:
    active = normalized_controls(controls)
    score, _ = answer_adequacy_score(answer, query, context, draft)
    reasons = []
    query_terms = set(re.findall(r"[a-zA-Z']+", (query or "").lower()))
    sensitive_terms = sorted(query_terms & REVIEW_TERMS)
    threshold = float(active["adequacy_threshold"])

    if score < threshold + 0.08:
        reasons.append(f"adequacy score {score:.2f} is close to review threshold {threshold:.2f}")
    if sentiment.get("label") == "negative":
        reasons.append("question sentiment is negative or urgent")
    if compute_state.get("compute_result") is not None:
        reasons.append("answer includes deterministic calculation output")
    if sensitive_terms:
        reasons.append(f"sensitive policy area: {', '.join(sensitive_terms)}")

    return score, reasons if active.get("human_review") else []
