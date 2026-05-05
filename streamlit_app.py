from pathlib import Path
import importlib.util
import sys
import types

from aegis_enhancements import apply
from deepseek_provider import apply_deepseek


# Streamlit Cloud may have an OPENAI_API_KEY secret with no embedding quota.
# Keep answer-generation keys usable, but force indexing to stay local-first
# from the deployed entrypoint so quota/rate-limit errors cannot crash startup.
if "langchain_openai" not in sys.modules:
    sys.modules["langchain_openai"] = types.ModuleType("langchain_openai")

app_path = Path(__file__).with_name("app.py")
spec = importlib.util.spec_from_file_location("aegis_app", app_path)
aegis_app = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(aegis_app)
apply(aegis_app)
apply_deepseek(aegis_app)
aegis_app.main()
