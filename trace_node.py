"""Trace Node — persists execution trace to disk."""
import json, logging, os, pathlib, time
from app.state import AgentState
from app.utils.tracing import trace as _trace

logger    = logging.getLogger(__name__)
TRACE_DIR = pathlib.Path(os.getenv("TRACE_DIR", "/tmp/dg_rag_traces"))

def run(state: AgentState) -> AgentState:
    record = {
        "ts": time.time(), "query": state.get("query"), "route": state.get("route"),
        "grade": state.get("employee_grade"), "verified": state.get("verified"),
        "hitl": state.get("hitl_decision"), "retries": state.get("retry_count", 0),
        "tokens": state.get("token_count", 0), "steps": state.get("trace_log", []),
    }
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        (TRACE_DIR / f"{int(time.time()*1000)}.json").write_text(json.dumps(record, indent=2, default=str))
    except Exception as e:
        logger.warning("Trace persist failed: %s", e)
    logger.info("Trace done — route=%s verified=%s retries=%d", record["route"], record["verified"], record["retries"])
    return _trace(state, node="trace", data={"steps": len(record["steps"])})
