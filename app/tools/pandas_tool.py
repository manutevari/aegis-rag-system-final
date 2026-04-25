from langchain.tools import tool
import pandas as pd

@tool
def pandas_query(csv_path: str, operation: str) -> str:
    """
    Execute dataframe operations on CSV data.
    """
    try:
        df = pd.read_csv(csv_path)

        # VERY IMPORTANT: controlled eval
        result = eval(operation, {"df": df, "pd": pd})

        return str(result)

    except Exception as e:
        return f"Error in pandas_query: {str(e)}"
