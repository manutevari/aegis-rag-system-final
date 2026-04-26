"""
Agent State — Unified one-shot definition with Pydantic
"""

from typing import Any, Dict, List, Union
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    # ── Input ──
    query: str
    history: List[Dict[str, Any]] = Field(default_factory=list)
    employee_grade: str = Field(default="L3")

    # ── Routing ──
    route: str = Field(default="retrieval")  # sql | retrieval | compute | direct

    # ── Retrieval ──
    retrieval_docs: List[str] = Field(default_factory=list)
    context: str = Field(default="")
    sources: List[str] = Field(default_factory=list)

    # ── Compute ──
    compute_result: float | None = None
    compute_steps: List[str] = Field(default_factory=list)
    compute_summary: str = Field(default="")

    # ── Generation ──
    answer: str = Field(default="")
    raw_answer: str = Field(default="")

    # ── Validation ──
    verified: bool = False
    verification_checks: Dict[str, bool] = Field(default_factory=dict)

    # ── Retry Control ──
    retry_count: int = 0
    max_retries: int = 3

    # ── HITL ──
    hitl_mode: str = Field(default="auto")      # auto | queue | cli
    hitl_decision: str = Field(default="approve")  # approve | reject | edit
    hitl_feedback: str = Field(default="")

    # ── Logging ──
    trace_log: List[Dict[str, Any]] = Field(default_factory=list)
    error: str = Field(default="")

    class Config:
        arbitrary_types_allowed = True

# 🔹 Helper: accept dict or AgentState seamlessly
def to_state(data: Union[Dict[str, Any], AgentState]) -> AgentState:
    if isinstance(data, AgentState):
        return data
    return AgentState(**data)

# 🔹 Example usage
if __name__ == "__main__":
    # dict input
    raw = {"query": "Laptop budget for L6?", "employee_grade": "L6"}
    state = to_state(raw)
    print(state.dict())

    # model input
    model_state = AgentState(query="Per diem for L4?")
    print(model_state.dict())
