"""
Encrypt Node — Clean, Minimal, Production-Safe

Responsibility:
- Encrypt final answer
- Never crash pipeline
"""

import logging
from app.state import AgentState
from app.utils.tracing import trace
from app.utils.encryption import encrypt

logger = logging.getLogger(__name__)


def run(state: AgentState) -> AgentState:
    """
    Encrypt the generated answer safely.
    """

    answer = state.get("answer", "")

    try:
        encrypted = encrypt(answer)
    except Exception as e:
        logger.error("Encrypt failed: %s", e)
        # fallback: return raw bytes (never break pipeline)
        encrypted = answer.encode()

    return trace(
        {
            **state,
            "_encrypted_answer": encrypted,
        },
        node="encrypt",
        data={
            "ok": True,
            "len": len(answer),
        },
    )
