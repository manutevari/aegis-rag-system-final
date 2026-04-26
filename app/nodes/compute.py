"""
Compute Node — Thin orchestration layer.

Delegates ALL computation to app.tools.compute.compute().
No business logic here.
"""

import logging
from app.state import AgentState
from app.tools.compute import compute
from app.utils.tracing import trace

logger = logging.getLogger(__name__)


def run(state: AgentState) -> AgentState:
    result = compute(state)

    logger.info(
        "Compute — result=%s steps=%d",
        result.get("compute_result"),
        len(result.get("compute_steps", [])),
    )

    return trace(
        result,
        node="compute",
        data={
            "result": result.get("compute_result"),
            "steps": len(result.get("compute_steps", [])),
        },
    )
