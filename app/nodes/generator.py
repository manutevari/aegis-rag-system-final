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
- Use clear markdown formatting.
- Make numbers and key policy constraints easy to scan.
- End with a source line when source information is available.
"""


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


def safe_invoke_llm(messages, model_override=None):
    try:
        return invoke_llm(messages, model_override=model_override)
    except Exception as exc:
        err = str(exc)
        if "429" in err or "quota" in err.lower():
            logger.warning("[LLM] Rate limited")
        else:
            logger.error("[LLM] Error: %s", err)
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
                    f"QUESTION: {query}{grade_note}\n\n"
                    "Answer with markdown, source references, and exact values from context."
                ),
            }
        )

        response = safe_invoke_llm(messages, model_override=model_override)
        answer = getattr(response, "content", None)
        if not answer:
            try:
                answer = response.choices[0].message.content
            except Exception:
                answer = str(response)
        answer = _format_answer(answer.strip())

    except Exception as exc:
        logger.error("Generation error: %s", exc)
        answer = "System busy or rate-limited. Please try again."

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
        data={"len": len(answer), "model": model_override or "default"},
    )
