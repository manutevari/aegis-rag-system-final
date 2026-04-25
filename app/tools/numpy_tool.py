from langchain.tools import tool
import numpy as np
import math

@tool
def numpy_compute(expression: str) -> str:
    """
    MANDATORY for advanced numerical calculations.
    Supports arrays, statistics, vector ops.
    """
    try:
        allowed_names = {
            "np": np,
            "math": math
        }

        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)

    except Exception as e:
        return f"Error in numpy_compute: {str(e)}"
