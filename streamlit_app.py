from pathlib import Path
import runpy
import sys
import types


# Streamlit Cloud may have an OPENAI_API_KEY secret with no embedding quota.
# Keep answer-generation keys usable, but force indexing to stay local-first
# from the deployed entrypoint so quota/rate-limit errors cannot crash startup.
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = types.ModuleType("langchain_openai")

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
