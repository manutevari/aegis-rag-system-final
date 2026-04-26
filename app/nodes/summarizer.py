from app.core.models import get_llm

def run(state):
    text = state.get("context", "")

    if len(text) < 1000:
        return state

    llm = get_llm(model_override="gpt-4o-mini")

    summary = llm.invoke(f"Summarize:\n{text}")

    return {**state, "context": summary.content}
