"""
Generator Node — Strict, grounded LLM answer generation with beautiful formatting.
"""

import logging, re, time
from app.state import AgentState
from app.utils.tracing import trace
from app.core.models import invoke_llm

logger = logging.getLogger(__name__)

_SYSTEM = """You are a professional corporate policy assistant.

Answer ONLY using the provided policy context. Format your response beautifully.

Rules:
- Do not invent any values, policies, or assumptions.
- Use exact numbers from context (₹ / USD).
- If a computed value is present, use it exactly.
- Cite policy codes when available.
- If answer is missing, say: "This is not covered in the available policy data."

Format Requirements:
- Use markdown formatting (bold, bullet points, etc.)
- Use clear section headers with emojis
- Make numbers and key info stand out
- Use tables for comparisons
- End with: **Source:** [policy name/code] if available.

Example format:
## 📋 Policy Answer

**Key Details:**
- Point 1: Value
- Point 2: Value

**Amount:** ₹X,XXX or $X,XXX

**Eligibility:** Description

**Source:** Policy Code XYZ
"""


def _format_answer(raw_answer: str) -> str:
    """
    Beautify the LLM answer with markdown formatting.
    
    Args:
        raw_answer: Raw LLM response
    
    Returns:
        Formatted answer with markdown
    """
    if not raw_answer or raw_answer.startswith("⚠️"):
        return raw_answer
    
    # Already has good formatting
    if "##" in raw_answer or "**" in raw_answer or "- " in raw_answer:
        return raw_answer
    
    # Extract key sections
    lines = raw_answer.split("\n")
    formatted = []
    
    in_answer = False
    for line in lines:
        line = line.strip()
        
        if not line:
            formatted.append("")
            continue
        
        # Make amounts bold
        line = re.sub(r"(₹[\d,]+|USD [\d,]+|\$[\d,]+)", r"**\1**", line)
        
        # Make policy codes bold
        line = re.sub(r"(Policy|policy|Code|code):\s*([A-Z\d\-]+)", r"\1: **\2**", line)
        
        # Convert "Key:" patterns to bold
        if ":" in line and len(line) < 100:
            parts = line.split(":", 1)
            if len(parts) == 2:
                line = f"**{parts[0].strip()}:** {parts[1].strip()}"
        
        formatted.append(line)
    
    result = "\n".join(formatted)
    
    # Add header if missing
    if not result.startswith("#"):
        result = "## 📋 Policy Information\n\n" + result
    
    return result


# ✅ Minimal safe wrapper (no over-retry)
def safe_invoke_llm(messages, model_override=None):
    try:
        return invoke_llm(messages, model_override=model_override)
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            logger.warning("[LLM] Rate limited")
        else:
            logger.error(f"[LLM] Error: {err}")
        raise


def run(state: AgentState) -> AgentState:
    query   = state.get("query", "")
    context = state.get("context", "")
    history = state.get("history") or []
    grade   = state.get("employee_grade", "")
    model_override = state.get("model")

    grade_note = f" (Employee grade: {grade})" if grade else ""

    try:
        msgs = [{"role": "system", "content": _SYSTEM}]
        msgs += history[-4:]
        msgs.append({
            "role": "user",
            "content": f"POLICY CONTEXT:\n{context}\n\nQUESTION: {query}{grade_note}\n\nFormat your answer beautifully with markdown. Include policy codes and exact amounts."
        })

        response = safe_invoke_llm(msgs, model_override=model_override)

        # ✅ Robust extraction
        answer = getattr(response, "content", None)
        if not answer:
            try:
                answer = response.choices[0].message.content
            except Exception:
                answer = str(response)

        answer = answer.strip()
        
        # ✅ Format the answer beautifully
        answer = _format_answer(answer)

    except Exception as e:
        logger.error("Generation error: %s", e)
        answer = "⚠️ System busy or rate-limited. Please try again."

    # ✅ Extract sources
    sources = re.findall(r"(?:Source|source|CODE|Code):\s*\*?\*?([^\n*]+)\*?\*?", answer)
    sources = [s.strip() for s in sources if s.strip()]

    return trace(
        {
            **state,
            "answer": answer,
            "sources": sources or [],
            "retry": False  # ✅ HARD STOP (no loop)
        },
        node="generate",
        data={
            "len": len(answer),
            "model": model_override or "default"
        }
    )
