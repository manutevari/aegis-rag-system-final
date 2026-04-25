import logging
import os
import uuid
import pandas as pd
import matplotlib

# 🔥 REQUIRED for server environments (Streamlit/Docker)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from app.state import AgentState

logger = logging.getLogger(__name__)


def run(state: AgentState) -> AgentState:
    try:
        csv_path = state.get("csv_path")

        # ✅ Validate file path
        if not csv_path or not os.path.exists(csv_path):
            return {
                **state,
                "tool_used": "plot_chart",
                "tool_error": f"CSV not found: {csv_path}"
            }

        # ✅ Load data
        df = pd.read_csv(csv_path)
        cols = df.columns.tolist()

        # ✅ Validate columns
        if len(cols) < 2:
            return {
                **state,
                "tool_used": "plot_chart",
                "tool_error": "Need at least 2 columns to plot"
            }

        x_col, y_col = cols[0], cols[1]

        # ✅ Generate plot
        plt.figure()
        plt.plot(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        # ✅ Save safely in temp directory
        filename = f"/tmp/plot_{uuid.uuid4().hex}.png"
        plt.savefig(filename)
        plt.close()

        return {
            **state,
            "tool_used": "plot_chart",
            "plot_path": filename,
            "tool_output": f"Plot saved at {filename}"
        }

    except Exception as e:
        logger.error("Plot node error: %s", e)

        return {
            **state,
            "tool_used": "plot_chart",
            "tool_error": str(e)
        }
