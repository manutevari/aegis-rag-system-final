"""
AEGIS Streamlit bootstrap hooks.

Python imports sitecustomize automatically when it is available on sys.path.
This keeps the Tavily controls available without disturbing the existing app
entrypoint.
"""

from __future__ import annotations

import os


def _install_tavily_controls() -> None:
    try:
        import streamlit as st
        from app.tools.tavily_search import TAVILY_ENV
    except Exception:
        return

    original_set_page_config = st.set_page_config

    def secret_or_env(names) -> str:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        try:
            for name in names:
                value = st.secrets.get(name, "")
                if value:
                    return value
        except Exception:
            pass
        return ""

    def patched_set_page_config(*args, **kwargs):
        result = original_set_page_config(*args, **kwargs)
        try:
            with st.sidebar.expander("Search Tools", expanded=False):
                st.toggle(
                    "Use Tavily Web Search",
                    value=bool(secret_or_env(TAVILY_ENV)),
                    key="tavily_enabled",
                    help="Adds Tavily web search as supplementary evidence for the policy answer composer.",
                )
                st.text_input(
                    "Tavily API Key",
                    type="password",
                    key="tavily_api_key",
                    help=f"Uses {' or '.join(TAVILY_ENV)} if this field is blank.",
                )
                st.selectbox(
                    "Tavily Search Depth",
                    ["basic", "fast", "advanced", "ultra-fast"],
                    index=0,
                    key="tavily_search_depth",
                )
                st.slider(
                    "Tavily Max Results",
                    min_value=1,
                    max_value=10,
                    value=3,
                    key="tavily_max_results",
                )
        except Exception:
            pass
        return result

    st.set_page_config = patched_set_page_config


_install_tavily_controls()
