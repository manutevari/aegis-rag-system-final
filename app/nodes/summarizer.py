from app.core.models import get_llm


def run(state):
    text = state.get("context", "")

    if len(text) < 1000:
        return state

    llm = get_llm(node="summarizer")
    summary = llm.invoke(f"Summarize:\n{text}")

    return {**state, "context": summary.content}
