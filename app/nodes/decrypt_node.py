"""Decrypt Node"""
import logging
from app.state import AgentState
from app.utils.encryption import decrypt
from app.utils.tracing import trace
logger = logging.getLogger(__name__)

def run(state: AgentState) -> AgentState:
    enc = state.get("_encrypted_answer", b"")
    try:    answer = decrypt(enc) if isinstance(enc, bytes) else state.get("answer","")
    except: answer = state.get("answer","")
    return trace({**state, "answer": answer}, node="decrypt", data={"ok": True})
