import logging
import pandas as pd
import matplotlib

# 🔥 REQUIRED for deployment (no display server)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import uuid
import os

from app.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> AgentState:
    try:
        csv_path = state.get("csv_path", "data.csv")

        if not os.path.exists(csv_path):
            return {
                **state,
                "tool_used": "plot_chart",
                "tool_error": f"CSV not found: {csv_path}"
            }

        df = pd.read_csv(csv_path)

        # Default columns (you can improve later)
        cols = list(df.columns)

        if len(cols) < 2:
            return {
                **state,
                "tool_used": "plot_chart",
                "tool_error": "Not enough columns to plot"
            }

        x_col, y_col = cols[0], cols[1]

        plt.figure()
        plt.plot(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        filename = f"plot_{uuid.uuid4().hex}.png"
        plt.savefig(filename)
        plt.close()

        return {
            **state,
            "tool_used": "plot_chart",
            "plot_path": filename,
            "tool_output": f"Plot saved: {filename}"
        }

    except Exception as e:
        logger.error("Plot node error: %s", e)

        return {
            **state,
            "tool_used": "plot_chart",
            "tool_error": str(e)
        }
