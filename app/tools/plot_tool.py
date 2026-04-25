from langchain.tools import tool
import matplotlib.pyplot as plt
import pandas as pd
import uuid

@tool
def plot_chart(csv_path: str, x_col: str, y_col: str) -> str:
    """
    Generate a simple line plot and save as image.
    """
    try:
        df = pd.read_csv(csv_path)

        plt.figure()
        plt.plot(df[x_col], df[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)

        filename = f"plot_{uuid.uuid4().hex}.png"
        plt.savefig(filename)
        plt.close()

        return f"Plot saved as {filename}"

    except Exception as e:
        return f"Error in plot_chart: {str(e)}"
