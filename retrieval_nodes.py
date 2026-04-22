def node_generate(state: AegisState) -> dict:
    """
    Build the final grounded answer using LCEL.
    Enforces strict token limits and performs post-generation hallucination checks.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser

    query  = state.query
    chunks = state.reranked_chunks

    # ── 1. Fallback: No Context ──────────────────────────────────────────
    if not chunks:
        answer = "I could not find relevant policy information. Please check the document ingestion status."
        log = tool_log(tool_name="generate", reason="Zero chunks available", inputs={"query": query})
        return {"final_answer": answer, "messages": [log]}

    # ── 2. Token Budgeting (Character-based heuristic) ───────────────────
    max_chars = state.max_context_tokens * 4
    used_chars = 0
    lc_docs = []

    for c in chunks:
        if used_chars + len(c.chunk_text) > max_chars:
            break
        lc_docs.append(Document(
            page_content=c.chunk_text,
            metadata={"section": f"{c.h1_header} > {c.h2_header}"}
        ))
        used_chars += len(c.chunk_text)

    # ── 3. LCEL Chain Execution ──────────────────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", _GEN_SYSTEM),
        ("human", "{input}")
    ])
    
    context_text = "\n\n".join([f"[{d.metadata['section']}]: {d.page_content}" for d in lc_docs])
    chain = prompt | _llm() | StrOutputParser()

    try:
        answer = chain.invoke({"input": query, "context": context_text})
        ai_msg = AIMessage(content=answer)
    except Exception as exc:
        answer = f"Generation Error: {exc}"
        ai_msg = AIMessage(content=answer)

    # ── 4. Hallucination Risk Assessment ────────────────────────────────
    context_texts = [d.page_content for d in lc_docs]
    is_risky, score = check_hallucination_risk(answer, context_texts)

    if is_risky:
        fb = handle_fallback(FallbackReason.HALLUCINATION, "node_generate", query=query)
        answer, ai_msg = fb.answer, AIMessage(content=fb.answer)

    return {
        "final_answer": answer,
        "hallucination_risk": is_risky,
        "messages": [ai_msg, tool_log("generate", "Grounded answer produced", {"risk_score": score})]
    }
