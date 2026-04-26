# ==============================
# ✅ SAFE + COMPLETE NODE EXPORT
# ==============================

from . import planner
from . import generator
from . import verifier

# -------- Optional Nodes (guarded) --------
def _safe_import(name):
    try:
        module = __import__(f"app.nodes.{name}", fromlist=[name])
        return module
    except Exception:
        return None

retrieval = _safe_import("retrieval")
sql_tool = _safe_import("sql_tool")
compute = _safe_import("compute")
context_assembler = _safe_import("context_assembler")
token_manager = _safe_import("token_manager")
hitl = _safe_import("hitl")
encrypt_node = _safe_import("encrypt_node")
decrypt_node = _safe_import("decrypt_node")
trace_node = _safe_import("trace_node")

# -------- New Nodes --------
numpy_node = _safe_import("numpy_node")
pandas_node = _safe_import("pandas_node")
plot_node = _safe_import("plot_node")


# ==============================
# ✅ EXPORT CONTROL (IMPORTANT)
# ==============================

__all__ = [
    "planner",
    "generator",
    "verifier",
    "retrieval",
    "sql_tool",
    "compute",
    "context_assembler",
    "token_manager",
    "hitl",
    "encrypt_node",
    "decrypt_node",
    "trace_node",
    "numpy_node",
    "pandas_node",
    "plot_node",
]
