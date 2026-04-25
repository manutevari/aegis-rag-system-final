import logging
import numpy as np
import math

from app.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")

    try:
        # Simple safe eval environment
        allowed = {
            "np": np,
            "math": math
        }

        # Extract expression (basic fallback)
        expression = query.lower().replace("calculate", "").strip()

        result = eval(expression, {"__builtins__": {}}, allowed)

        return {
            **state,
            "tool_used": "numpy_compute",
            "tool_input": expression,
            "tool_output": str(result)
        }

    except Exception as e:
        logger.error("NumPy node error: %s", e)

        return {
            **state,
            "tool_used": "numpy_compute",
            "tool_error": str(e)
        }
