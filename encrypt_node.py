"""Encrypt Node"""
import logging
from app.state import AgentState
from app.utils.encryption import encrypt
from app.utils.tracing import trace
logger = logging.getLogger(__name__)

def run(state: AgentState) -> AgentState:
    try:    enc = encrypt(state.get("answer", ""))
    except Exception as e:
        logger.error("Encrypt failed: %s", e); enc = state.get("answer","").encode()
    return trace({**state, "_encrypted_answer": enc}, node="encrypt", data={"ok": True})
