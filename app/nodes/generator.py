"""Generator node for grounded policy answers."""

import logging
import re

from app.core.models import invoke_llm
from app.core.stability_patch import safe_get, with_updates
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)

NO_POLICY_DATA_ANSWER = "No policy data found. Please refine your query."

_SYSTEM = """You are a professional corporate policy assistant.

Answer only using the provided policy context.

Rules:
- Do not invent values, policies, or assumptions.
- Use exact numbers from context.
- Cite policy codes or source files when available.
- If the answer is missing, say: "This is not covered in the available policy data."

Format Requirements:
- Use clear, polished markdown formatting.
- Adapt the answer tone to the user sentiment supplied in the prompt.
- Make numbers and key policy constraints easy to scan.
- Include deterministic calculation results exactly when supplied.
- Include a brief "Reasoning" section with source-grounded evidence, not hidden chain-of-thought.
- End with a source line when source information is available.
"""

_POSITIVE_WORDS = {"thanks", "thank", "great", "good", "happy", "excellent", "helpful", "clear", "perfect"}
_NEGATIVE_WORDS = {
    "angry",
    "bad",
    "confused",
    "denied",
    "frustrated",
    "issue",
    "problem",
    "reject",
    "rejected",
    "urgent",
    "worried",
    "wrong",
}


def _sentiment_profile(query: str, state: AgentState) -> dict:
    label = safe_get(state, "sentiment_label")
    tone = safe_get(state, "sentiment_tone")
    if label and tone:
        return {"label": label, "tone": tone}

    words = re.findall(r"[a-zA-Z']+", (query or "").lower())
    positive = sum(1 for word in words if word in _POSITIVE_WORDS)
    negative = sum(1 for word in words if word in _NEGATIVE_WORDS)
    if negative > positive:
        return {"label": "negative", "tone": "empathetic, calm, and reassuring"}
    if positive > negative:
        return {"label": "positive", "tone": "warm, confident, and concise"}
    return {"label": "neutral", "tone": "clear, professional, and human"}


def _format_answer(raw_answer: str) -> str:
    if not raw_answer:
        return raw_answer
    if raw_answer.startswith("No policy data found"):
        return raw_answer
    if "##" in raw_answer or "**" in raw_answer or "- " in raw_answer:
        return raw_answer

    lines = []
    for line in raw_answer.split("\n"):
        line = line.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"(INR [\d,]+|USD [\d,]+|\$[\d,]+)", r"**\1**", line)
        if ":" in line and len(line) < 100:
            key, value = line.split(":", 1)
            line = f"**{key.strip()}:** {value.strip()}"
        lines.append(line)

    result = "\n".join(lines)
    if not result.startswith("#"):
        result = "## Policy Information\n\n" + result
    return result


def safe_invoke_llm(messages, model_override=None, node="generator"):
    try:
        return invoke_llm(messages, model_override=model_override, node=node)
    except Exception as exc:
        logger.error("[Local LLM] Error: %s", exc)
        raise


def _has_policy_documents(state: AgentState) -> bool:
    return bool(safe_get(state, "documents") or safe_get(state, "retrieval_docs"))


def _sources_from_documents(state: AgentState):
    sources = []
    for doc in safe_get(state, "documents", []) or []:
        source = None
        if isinstance(doc, dict):
            source = doc.get("source")
            metadata = doc.get("metadata") or {}
            source = source or metadata.get("source") or metadata.get("source_path")
        else:
            metadata = getattr(doc, "metadata", {}) or {}
            source = metadata.get("source") or metadata.get("source_path")
        if source and source not in sources:
            sources.append(source)
    return sources


def run(state: AgentState) -> AgentState:
    query = safe_get(state, "query", "") or ""
    context = safe_get(state, "context", "") or ""
    history = safe_get(state, "history") or []
    grade = safe_get(state, "employee_grade", "") or ""
    model_override = safe_get(state, "model")
    model_decision = {"provider": "extractive", "model": "extractive"}
    sentiment = _sentiment_profile(query, state)
    compute_summary = safe_get(state, "compute_summary", "") or ""

    if not _has_policy_documents(state):
        return trace(
            with_updates(
                state,
                answer=NO_POLICY_DATA_ANSWER,
                response=NO_POLICY_DATA_ANSWER,
                sources=[],
                retry=False,
            ),
            node="generate",
            data={"len": len(NO_POLICY_DATA_ANSWER), "guard": "no_documents"},
        )

    grade_note = f" (Employee grade: {grade})" if grade else ""

    try:
        messages = [{"role": "system", "content": _SYSTEM}]
        messages += history[-4:]
        messages.append(
            {
                "role": "user",
                "content": (
                    f"POLICY CONTEXT:\n{context}\n\n"
                    f"SENTIMENT: {sentiment['label']}\n"
                    f"RESPONSE TONE: {sentiment['tone']}\n"
                    f"DETERMINISTIC CALCULATION:\n{compute_summary or 'No deterministic calculation supplied.'}\n\n"
                    f"QUESTION: {query}{grade_note}\n\n"
                    "Answer with markdown, source references, exact values from context, "
                    "and a short source-grounded Reasoning section."
                ),
            }
        )

        response = safe_invoke_llm(messages, model_override=model_override, node="generator")
        model_decision = getattr(response, "model_decision", model_decision) or model_decision
        answer = getattr(response, "content", None)
        if not answer:
            try:
                answer = response.choices[0].message.content
            except Exception:
                answer = str(response)
        answer = _format_answer(answer.strip())

    except Exception as exc:
        logger.error("Generation error: %s", exc)
        answer = "Local generation failed. Please check the configured local model runtime and try again."

    sources = re.findall(r"(?:Source|source|CODE|Code):\s*\*?\*?([^\n*]+)\*?\*?", answer)
    sources = [source.strip() for source in sources if source.strip()]
    sources = sources or _sources_from_documents(state)

    return trace(
        with_updates(
            state,
            answer=answer,
            response=answer,
            sources=sources,
            retry=False,
        ),
        node="generate",
        data={"len": len(answer), "model_decision": model_decision, "sentiment": sentiment},
    )
