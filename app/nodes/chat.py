from app.core.models import get_llm
from langchain_core.prompts import ChatPromptTemplate

def run(state: dict):
    # 1. Initialize Fast LLM for Query Transformation [cite: 42]
    # Using gpt-4o-mini as a high-speed orchestrator
    llm_fast = get_llm(model_override="gpt-4o-mini", temperature=0)
    
    raw_query = state["query"]
    
    # 2. Step 1: Query Transformation (Expansion) [cite: 40, 42]
    # We generate multiple variations to ensure we don't miss relevant chunks 
    # due to "lazy prompt writing"[cite: 38].
    expansion_prompt = ChatPromptTemplate.from_template("""
        You are an AI language optimizer. Generate 3 diverse variations of the 
        user's query to improve vector database retrieval.
        User Input: {query}
        Provide only the queries, one per line.
    """)
    
    expansion_chain = expansion_prompt | llm_fast
    expansion_res = expansion_chain.invoke({"query": raw_query})
    queries = [raw_query] + expansion_res.content.strip().split("\n")

    # 3. Step 2: Metadata Intent Extraction (Pre-Filtering) [cite: 51, 53]
    # Determine the 'policy_category' to prevent mathematically incorrect context[cite: 55].
    intent_prompt = ChatPromptTemplate.from_template("""
        Classify the query into a policy category: (Travel, HR, IT, Finance, or General).
        Query: {query}
        Category:
    """)
    intent_chain = intent_prompt | llm_fast
    category_intent = intent_chain.invoke({"query": raw_query}).content.strip()

    # 4. Preparation for Retrieval & Reranking [cite: 58, 62]
    # We pass the expanded queries and the metadata filter to the next node (Retriever)
    return {
        **state,
        "expanded_queries": queries,
        "search_filters": {"policy_category": category_intent},
        "step": "retrieval_ready"
    }
