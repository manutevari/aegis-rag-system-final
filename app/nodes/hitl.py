"""
HITL Node — Human review gate.
Modes: "auto" (skip), "cli" (stdin), "queue" (file-based, integrates with API).
"""
import json, logging, os, pathlib, time, uuid
from app.state import AgentState
from app.utils.tracing import trace

logger  = logging.getLogger(__name__)
MODE    = os.getenv("HITL_MODE", "auto")
TIMEOUT = int(os.getenv("HITL_TIMEOUT", "120"))
QUEUE   = pathlib.Path(os.getenv("HITL_QUEUE_DIR", "/tmp/hitl_queue"))
AUTO_PASS = os.getenv("HITL_AUTO_PASS", "true").lower() == "true"


def _cli(state: AgentState) -> dict:
    print("\n" + "="*60 + "\n🔍 HUMAN REVIEW\n" + "="*60)
    print(f"QUERY : {state.get('query')}")
    print(f"ANSWER:\n{state.get('answer')}")
    if state.get("verification_issues"):
        print(f"ISSUES: {'; '.join(state['verification_issues'])}")
    c = input("Decision [(a)pprove/(e)dit/(r)eject]: ").strip().lower()
    if c.startswith("e"):
        return {"status": "edited", "edited_answer": input("Corrected answer:\n> ").strip()}
    if c.startswith("r"):
        return {"status": "rejected", "edited_answer": None}
    return {"status": "approved", "edited_answer": None}


def _queue(state: AgentState, rid: str) -> dict:
    QUEUE.mkdir(parents=True, exist_ok=True)
    item = {"review_id": rid, "query": state.get("query"), "answer": state.get("answer"),
            "issues": state.get("verification_issues", []), "status": "pending"}
    (QUEUE / f"{rid}.json").write_text(json.dumps(item, indent=2))
    deadline = time.time() + TIMEOUT
    while time.time() < deadline:
        try:
            data = json.loads((QUEUE / f"{rid}.json").read_text())
            if data.get("status") != "pending":
                return data
        except: pass
        time.sleep(1)
    return {"status": "approved" if AUTO_PASS else "rejected", "edited_answer": None}


def run(state: AgentState) -> AgentState:
    rid = str(uuid.uuid4())[:8]
    if   MODE == "cli":   decision = _cli(state)
    elif MODE == "queue": decision = _queue(state, rid)
    else:                 decision = {"status": "approved", "edited_answer": None}

    status = decision.get("status", "approved")
    edited = decision.get("edited_answer")
    hitl_decision = "reject" if status == "rejected" else ("edit" if edited else "approve")
    final = edited if edited else state.get("answer", "")

    logger.info("HITL decision=%s rid=%s", hitl_decision, rid)
    return trace({**state, "answer": final, "hitl_decision": hitl_decision, "hitl_edited_answer": edited},
                 node="hitl", data={"decision": hitl_decision, "review_id": rid})
