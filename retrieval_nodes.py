def node_generate(state: AegisState) -> dict:
    """
    Hybrid optimized generation node:
    - Clean LCEL execution
    - Token-safe context handling
    - Minimal but essential logging
    - Strong hallucination protection
    """

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser

    query  = state.query
    chunks = state.reranked_chunks

    # ── 1. Fallback (strict + aligned with system prompt) ───────────────
    if not chunks:
        answer = (
            "Not found in context. Please check that the relevant policy document has been ingested."
        )
        return {
            "final_answer": answer,
            "hallucination_risk": False,
            "messages": [
                tool_log("generate", "No chunks available", {"query": query})
            ],
        }

    # ── 2. Token Budgeting (with partial salvage) ───────────────────────
    max_chars   = state.max_context_tokens * 4
    used_chars  = 0
    lc_docs     = []

    for c in chunks:
        chunk_len = len(c.chunk_text)

        if used_chars + chunk_len > max_chars:
            remaining = max_chars - used_chars
            if remaining > 200:
                lc_docs.append(Document(
                    page_content=c.chunk_text[:remaining] + "...",
                    metadata={
                        "section": f"{c.h1_header} > {c.h2_header}",
                        "document_id": c.document_id,
                        "category": c.policy_category,
                    },
                ))
            break

        lc_docs.append(Document(
            page_content=c.chunk_text,
            metadata={
                "section": f"{c.h1_header} > {c.h2_header}",
                "document_id": c.document_id,
                "category": c.policy_category,
            },
        ))
        used_chars += chunk_len

    # ── 3. Context Construction (structured + readable) ────────────────
    context_text = "\n\n".join([
        f"[{d.metadata['section']} | {d.metadata['document_id']}]: {d.page_content}"
        for d in lc_docs
    ])

    # ── 4. LCEL Chain (modern LangChain) ───────────────────────────────
    prompt = ChatPromptTemplate.from_messages([
        ("system", _GEN_SYSTEM),
        ("human", "{input}")
    ])

    chain = prompt | _llm() | StrOutputParser()

    try:
        answer = chain.invoke({
            "input": query,
            "context": context_text
        })
        ai_msg = AIMessage(content=answer)

    except Exception as exc:
        answer = f"Generation Error: {exc}"
        ai_msg = AIMessage(content=answer)

    # ── 5. Hallucination Check (essential safety) ───────────────────────
    context_texts = [d.page_content for d in lc_docs]
    is_risky, score = check_hallucination_risk(answer, context_texts)

    if is_risky:
        fb = handle_fallback(
            FallbackReason.HALLUCINATION,
            "node_generate",
            query=query,
            detail=f"score={score:.2%}"
        )
        answer = fb.answer
        ai_msg = AIMessage(content=answer)

    # ── 6. Minimal Observability (balanced) ─────────────────────────────
    log = tool_log(
        tool_name="generate",
        reason="Final answer generation (LCEL hybrid)",
        inputs={"query": query},
        outputs={
            "answer_len": len(answer),
            "chunks_used": len(lc_docs),
            "hallucination_risk": is_risky,
            "score": score,
        },
    )

    return {
        "final_answer": answer,
        "hallucination_risk": is_risky,
        "messages": [ai_msg, log],
    }
