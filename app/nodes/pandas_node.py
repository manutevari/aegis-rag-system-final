import logging
import pandas as pd
import os

from app.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> AgentState:
    query = state.get("query", "")

    try:
        # ⚠️ You MUST define how CSV path is passed
        csv_path = state.get("csv_path", "data.csv")

        if not os.path.exists(csv_path):
            return {
                **state,
                "tool_used": "pandas_query",
                "tool_error": f"CSV not found: {csv_path}"
            }

        df = pd.read_csv(csv_path)

        # Safe operation filter
        ALLOWED = ["mean", "sum", "groupby", "max", "min"]

        operation = query.lower()

        if not any(k in operation for k in ALLOWED):
            return {
                **state,
                "tool_used": "pandas_query",
                "tool_error": "Operation not allowed"
            }

        result = eval(operation, {"df": df, "pd": pd})

        return {
            **state,
            "tool_used": "pandas_query",
            "tool_output": str(result)
        }

    except Exception as e:
        logger.error("Pandas node error: %s", e)

        return {
            **state,
            "tool_used": "pandas_query",
            "tool_error": str(e)
        }
