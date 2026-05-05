"""
AEGIS Streamlit bootstrap hooks.

Python imports sitecustomize automatically when it is available on sys.path.
These hooks keep user-facing controls near the top of the app while allowing
the existing Streamlit entrypoint to stay small.
"""

from __future__ import annotations

import inspect
import os


def _install_aegis_controls() -> None:
    try:
        import streamlit as st
        from app.core import dynamic_orchestration as orchestration
        from app.tools.tavily_search import TAVILY_ENV
    except Exception:
        return

    if getattr(st, "_aegis_controls_installed", False):
        return
    st._aegis_controls_installed = True

    original_set_page_config = st.set_page_config
    original_chat_input = st.chat_input
    original_info = st.info
    original_warning = st.warning
    original_success = st.success

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

    def controls() -> dict:
        return orchestration.normalized_controls(
            {
                "mode": st.session_state.get("orchestration_mode", "Balanced"),
                "adequacy_threshold": st.session_state.get("orchestration_adequacy_threshold", 0.52),
                "max_cloud_attempts": st.session_state.get("orchestration_max_cloud_attempts", 3),
                "human_review": st.session_state.get("orchestration_human_review", True),
            }
        )

    def patch_app_globals(frame) -> None:
        if not frame:
            return
        globals_map = frame.f_globals
        if globals_map.get("_AEGIS_DYNAMIC_ORCHESTRATION_PATCHED"):
            return
        required = {"_CLOUD_PROVIDERS", "_provider_key"}
        if not required.issubset(globals_map):
            return

        cloud_providers = globals_map["_CLOUD_PROVIDERS"]
        provider_key = globals_map["_provider_key"]

        def rank_auto_providers(sentiment, compute_state):
            return orchestration.rank_providers(sentiment, compute_state, controls())

        def provider_candidate_chain(selected_provider, sentiment, compute_state):
            return orchestration.candidate_chain(
                selected_provider,
                sentiment,
                compute_state,
                cloud_providers,
                provider_key,
                controls(),
            )

        def answer_is_relevant(answer, query, context, draft):
            return orchestration.answer_is_relevant(answer, query, context, draft, controls())

        globals_map["rank_auto_providers"] = rank_auto_providers
        globals_map["provider_candidate_chain"] = provider_candidate_chain
        globals_map["answer_is_relevant"] = answer_is_relevant
        globals_map["_AEGIS_DYNAMIC_ORCHESTRATION_PATCHED"] = True

    def patched_set_page_config(*args, **kwargs):
        result = original_set_page_config(*args, **kwargs)
        try:
            with st.sidebar.expander("Human Oversight & Orchestration", expanded=False):
                st.selectbox(
                    "Routing Strategy",
                    ["Balanced", "Cost efficient", "Highest adequacy"],
                    index=0,
                    key="orchestration_mode",
                )
                st.slider(
                    "Minimum Adequacy",
                    min_value=0.30,
                    max_value=0.90,
                    value=0.52,
                    step=0.01,
                    key="orchestration_adequacy_threshold",
                )
                st.slider(
                    "Max Hosted Attempts",
                    min_value=1,
                    max_value=6,
                    value=3,
                    key="orchestration_max_cloud_attempts",
                )
                st.toggle(
                    "Human Final Review",
                    value=True,
                    key="orchestration_human_review",
                )

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

    def patched_chat_input(*args, **kwargs):
        frame = inspect.currentframe()
        patch_app_globals(frame.f_back if frame else None)
        return original_chat_input(*args, **kwargs)

    def patched_info(body, *args, **kwargs):
        result = original_info(body, *args, **kwargs)
        if isinstance(body, str) and body.startswith("Tone:"):
            frame = inspect.currentframe()
            caller = frame.f_back if frame else None
            local_map = caller.f_locals if caller else {}
            query = local_map.get("query", "")
            sentiment = local_map.get("sentiment", {})
            compute_state = local_map.get("compute_state", {})
            final_answer = local_map.get("final_answer", "")
            result_payload = local_map.get("result", {}) or {}
            context = result_payload.get("context") or local_map.get("context", "")
            draft = result_payload.get("answer") or local_map.get("draft_answer", "")
            score, reasons = orchestration.human_review_reasons(
                query,
                sentiment,
                compute_state,
                final_answer,
                context,
                draft,
                controls(),
            )
            st.caption(f"Adequacy score: {score:.2f} | Routing: {controls()['mode']}")
            if reasons:
                original_warning("Human review recommended before acting on this answer.")
                with st.expander("Human Review Reasons", expanded=True):
                    for reason in reasons:
                        st.write(f"- {reason}")
                    st.checkbox("Reviewed by human", key=f"reviewed_{len(st.session_state.get('messages', []))}")
            elif controls().get("human_review"):
                original_success("Human oversight check passed.")
        return result

    st.set_page_config = patched_set_page_config
    st.chat_input = patched_chat_input
    st.info = patched_info


_install_aegis_controls()
