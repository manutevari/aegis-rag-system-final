"""Token Manager — guards against context overflow."""
import logging, os
from app.state import AgentState
from app.utils.tracing import trace

logger = logging.getLogger(__name__)
TOKEN_THRESHOLD = int(os.getenv("TOKEN_THRESHOLD", "3000"))

_SYS = """Compress the context to under 1500 tokens.
RULES: keep ALL numbers, rates, policy codes EXACTLY as-is. Remove only redundant prose."""

def run(state: AgentState) -> AgentState:
    return trace(state, node="token_check", data={"tokens": state.get("token_count", 0)})

def summarize(state: AgentState) -> AgentState:
    context = state.get("context", "")
    query   = state.get("query", "")
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0, max_tokens=1500)
        compressed = llm.invoke([
            {"role": "system", "content": _SYS},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUERY: {query}"},
        ]).content.strip()
    except Exception as e:
        logger.error("Summarisation failed, keeping original: %s", e)
        compressed = context
    new_tokens = len(compressed) // 4
    return trace({**state, "context": compressed, "token_count": new_tokens, "context_summarized": True},
                 node="summarize_context", data={"new_tokens": new_tokens})
