from app.core.models import get_llm
from langchain_core.prompts import ChatPromptTemplate


def run(state: dict):
    llm_fast = get_llm(node="chat", temperature=0)

    raw_query = state["query"]

    expansion_prompt = ChatPromptTemplate.from_template(
        """
        You are an AI language optimizer. Generate 3 diverse variations of the
        user's query to improve vector database retrieval.
        User Input: {query}
        Provide only the queries, one per line.
    """
    )

    expansion_chain = expansion_prompt | llm_fast
    expansion_res = expansion_chain.invoke({"query": raw_query})
    queries = [raw_query] + expansion_res.content.strip().split("\n")

    intent_prompt = ChatPromptTemplate.from_template(
        """
        Classify the query into a policy category: (Travel, HR, IT, Finance, or General).
        Query: {query}
        Category:
    """
    )
    intent_chain = intent_prompt | llm_fast
    category_intent = intent_chain.invoke({"query": raw_query}).content.strip()

    return {
        **state,
        "expanded_queries": queries,
        "search_filters": {"policy_category": category_intent},
        "step": "retrieval_ready",
    }
