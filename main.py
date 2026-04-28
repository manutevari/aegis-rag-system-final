"""Legacy entrypoint that runs the primary offline-first Streamlit app."""

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("streamlit_app.py")), run_name="__main__")
