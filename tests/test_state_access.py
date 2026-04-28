import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import app.nodes.router as router
from app.core.stability_patch import safe_get, safe_set
from app.nodes.router import run as route
from app.state import AgentState
from app.utils.tracing import trace


def test_agent_state_supports_mapping_get_and_item_access():
    state = AgentState(query="What is the travel policy?", employee_grade="L4")

    assert state.get("query") == "What is the travel policy?"
    assert state.get("missing", "fallback") == "fallback"
    assert state["employee_grade"] == "L4"

    state["intent"] = "rag"
    assert state.intent == "rag"


def test_agent_state_can_be_unpacked_as_mapping():
    state = AgentState(query="What is the travel policy?", trace_log=[])
    unpacked = {**state, "route": "retrieval"}

    assert unpacked["query"] == "What is the travel policy?"
    assert unpacked["route"] == "retrieval"


def test_trace_accepts_agent_state_object():
    state = AgentState(query="What is the travel policy?", trace_log=[])
    traced = trace(state, node="trace_start", data={"steps": 0})

    assert traced["query"] == "What is the travel policy?"
    assert traced["trace_log"][-1]["node"] == "trace_start"


def test_safe_access_helpers_work_for_dict_and_agent_state():
    raw = {"query": "calculate total", "route": "compute"}
    model = AgentState(query="policy allowance")

    assert safe_get(raw, "route") == "compute"
    assert safe_get(model, "query") == "policy allowance"

    safe_set(raw, "intent", "compute")
    safe_set(model, "intent", "rag")

    assert raw["intent"] == "compute"
    assert model.intent == "rag"


def test_router_accepts_agent_state_and_routes_policy_query():
    state = AgentState(query="Tell me about allowance eligibility", trace_log=[])
    routed = route(state)

    assert routed["intent"] == "rag"
    assert routed["route"] == "retrieval"


def test_router_handles_unclear_agent_state_without_crashing(monkeypatch):
    monkeypatch.setattr(router, "_llm_intent", lambda query: "unclear")

    state = AgentState(query="blue notebook", trace_log=[])
    routed = router.run(state)

    assert routed["intent"] == "rag"
    assert routed["route"] == "retrieval"
    assert any("fallback" in entry for entry in routed["trace_log"])
