"""
Agent state schema for the LangGraph pipeline.

The graph may hand nodes either a plain dict or this Pydantic model depending on
where execution enters. AgentState therefore exposes a small mapping-compatible
surface while keeping structured defaults for production runs.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    # Input
    query: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    memory_context: str = ""
    employee_grade: Optional[str] = "L3"

    # Routing
    route: str = "retrieval"
    intent: str = "rag"
    halted: bool = False

    # Retrieval/context
    documents: List[Any] = Field(default_factory=list)
    retrieval_docs: List[Any] = Field(default_factory=list)
    context: str = ""
    sources: List[str] = Field(default_factory=list)
    sql_result: List[Dict[str, Any]] = Field(default_factory=list)

    # Compute
    compute_result: Optional[float] = None
    compute_steps: List[str] = Field(default_factory=list)
    compute_summary: str = ""

    # Generation
    answer: str = ""
    raw_answer: str = ""

    # Validation
    verified: bool = False
    verification_checks: Dict[str, bool] = Field(default_factory=dict)
    confidence: float = 0.0

    # Token/accounting
    token_count: int = 0
    context_tokens: int = 0

    # Retry/HITL
    retry_count: int = 0
    max_retries: int = 3
    hitl_mode: str = "auto"
    hitl_decision: str = "approve"
    hitl_feedback: str = ""

    # Logging/error state
    trace_log: List[Any] = Field(default_factory=list)
    error: str = ""

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> "AgentState":
        setattr(self, key, value)
        return self

    def __getitem__(self, key: str) -> Any:
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def update(self, values: Optional[Dict[str, Any]] = None, **kwargs: Any) -> "AgentState":
        for key, value in {**(values or {}), **kwargs}.items():
            setattr(self, key, value)
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


def to_state(data: Union[Dict[str, Any], AgentState]) -> AgentState:
    if isinstance(data, AgentState):
        return data
    return AgentState(**data)


def state_to_dict(data: Union[Dict[str, Any], AgentState]) -> Dict[str, Any]:
    if isinstance(data, AgentState):
        return data.model_dump()
    return dict(data)
