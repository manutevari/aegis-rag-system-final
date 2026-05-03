from app.core.models import get_llm


def run(state: dict):
    llm_fast = get_llm(node="chat", temperature=0)
    raw_query = state["query"]

    expansion_res = llm_fast.invoke(
        [
            {
                "role": "user",
                "content": (
                    "Generate 3 diverse variations of the user's query to improve vector database retrieval.\n"
                    f"User Input: {raw_query}\n"
                    "Provide only the queries, one per line."
                ),
            }
        ]
    )
    queries = [raw_query] + expansion_res.content.strip().split("\n")

    category_res = llm_fast.invoke(
        [
            {
                "role": "user",
                "content": (
                    "Classify the query into a policy category: Travel, HR, IT, Finance, or General.\n"
                    f"Query: {raw_query}\nCategory:"
                ),
            }
        ]
    )
    category_intent = category_res.content.strip()

    return {
        **state,
        "expanded_queries": queries,
        "search_filters": {"policy_category": category_intent},
        "step": "retrieval_ready",
    }
