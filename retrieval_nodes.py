# =============================================================================
# AEGIS — LANGGRAPH RETRIEVAL NODES
# retrieval_nodes.py
#
# Each function is a LangGraph node: receives AegisState, returns partial dict.
# All state updates are Pydantic-validated before merge.
#
# Nodes (in graph execution order):
#   1. node_router          — deterministic keyword pre-check → LLM fallback
#   2. node_expand_query    — Multi-Query Expansion (LangChain LCEL)
#   3. node_hyde            — HyDE document generation (LangChain LCEL)
#   4. node_retrieve        — VectorStoreRetriever × N variants + dedup
#   5. node_post_filter     — drop stale policy versions by effective_date
#   6. node_rerank          — CrossEncoderReranker → Top-5
#   7. node_generate        — create_stuff_documents_chain → final answer
#
# Message protocol (every node appends to messages):
#   SystemMessage  — prompt / instruction context where relevant
#   HumanMessage   — user-facing input forwarded to LLM
#   AIMessage      — raw LLM text output
#   ToolMessage    — audit entry via tool_log() with reason + I/O
#
# Router logic:
#   KEYWORD_MAP gives deterministic O(1) classification for clear signals.
#   Only truly ambiguous queries fall through to the LLM classifier.
#   This keeps the router fast, predictable, and auditable.
# =============================================================================

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from graph_state import (
    AegisState,
    AIMessage,
    ChunkResult,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    VALID_CATEGORIES,
    tool_log,
)
from tfidf_index import (
    TfidfIndex,
    reciprocal_rank_fusion,
    tfidf_calibrate_scores,
)
from logger import get_logger, PipelineLogger
from fallback_handler import (
    FallbackReason,
    handle_fallback,
    check_hallucination_risk,
    NOT_FOUND_PHRASE,
)

# Module-level logger
_log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lazy clients
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0,
                      api_key=os.getenv("OPENAI_API_KEY", ""))

@lru_cache(maxsize=1)
def _llm_creative():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.3,
                      api_key=os.getenv("OPENAI_API_KEY", ""))

@lru_cache(maxsize=1)
def _embeddings():
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-large",
                            api_key=os.getenv("OPENAI_API_KEY", ""))

@lru_cache(maxsize=1)
def _vector_store():
    from langchain_pinecone import PineconeVectorStore
    return PineconeVectorStore(
        index_name="aegis-index",
        embedding=_embeddings(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
    )

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BROAD_K = 25
FINAL_K = 5

# Deterministic keyword → category map (lowercase tokens)
# Add/remove entries here — no LLM call consumed for these signals.
KEYWORD_MAP: dict[str, str] = {
    # Travel
    "travel": "Travel", "flight": "Travel", "hotel": "Travel",
    "per diem": "Travel", "taxi": "Travel", "uber": "Travel",
    "rideshare": "Travel", "mileage": "Travel", "fuel": "Travel",
    "expense": "Travel", "reimbursement": "Travel", "passport": "Travel",
    "visa": "Travel", "international": "Travel",
    # HR
    "leave": "HR", "maternity": "HR", "paternity": "HR", "pto": "HR",
    "vacation": "HR", "sick": "HR", "absence": "HR", "hr ": "HR",
    "performance": "HR", "compensation": "HR", "salary": "HR",
    "conduct": "HR", "disciplinary": "HR", "tuition": "HR",
    "learning": "HR", "training": "HR",
    # Finance
    "budget": "Finance", "finance": "Finance", "invoice": "Finance",
    "procurement": "Finance", "purchase": "Finance",
    # Legal
    "legal": "Legal", "contract": "Legal", "compliance": "Legal",
    "gdpr": "Legal", "privacy": "Legal",
    # IT
    "it security": "IT", "data security": "IT", "cybersecurity": "IT",
    "password": "IT", "vpn": "IT", "software": "IT", "device": "IT",
}

# ---------------------------------------------------------------------------
# Node 1 — Router
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM = (
    "You are a corporate policy intent classifier.\n"
    "Given a user question, return ONLY the single most relevant policy category\n"
    "from this list: Travel | HR | Finance | Legal | IT | General\n"
    "One word only. No punctuation. No explanation."
)


def node_router(state: AegisState) -> dict:
    """
    Classify query intent for Pinecone pre-filtering.

    Strategy (deterministic-first, LLM-fallback):
      1. Lowercase the query and scan KEYWORD_MAP.
         If any keyword matches → return immediately (confidence=high).
         No LLM call, no latency, fully auditable.
      2. If no keyword match → call LLM classifier (confidence=low).
         The LLM handles nuanced / multi-topic queries.
    """
    query      = state.query
    query_low  = query.lower()

    # ── Step 1: deterministic keyword scan ──────────────────────────────
    for kw, cat in KEYWORD_MAP.items():
        if kw in query_low:
            log = tool_log(
                tool_name="router",
                reason=f"Deterministic keyword match: '{kw}' → '{cat}'. "
                       "No LLM call needed; result is fully deterministic.",
                inputs={"query": query, "matched_keyword": kw},
                outputs={"category": cat, "confidence": "high"},
            )
            return {
                "detected_category": cat,
                "router_confidence":  "high",
                "messages": [log],
            }

    # ── Step 2: LLM classifier for ambiguous queries ─────────────────────
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", _ROUTER_SYSTEM),
        ("human",  "{query}"),
    ])
    chain = prompt | _llm() | StrOutputParser()

    sys_msg  = SystemMessage(content=_ROUTER_SYSTEM)
    user_msg = HumanMessage(content=query)

    try:
        raw = chain.invoke({"query": query})
        cat = raw.strip().title()
        cat = cat if cat in VALID_CATEGORIES else None
        ai_msg = AIMessage(content=raw)
    except Exception as exc:
        cat    = None
        ai_msg = AIMessage(content=f"router LLM failed: {exc}")

    log = tool_log(
        tool_name="router",
        reason="No deterministic keyword matched — falling back to LLM classification "
               "to handle ambiguous or multi-topic query.",
        inputs={"query": query},
        outputs={"category": cat, "confidence": "low"},
    )
    return {
        "detected_category": cat,
        "router_confidence":  "low",
        "messages": [sys_msg, user_msg, ai_msg, log],
    }

# ---------------------------------------------------------------------------
# Node 2 — Multi-Query Expansion
# ---------------------------------------------------------------------------

_EXPAND_SYSTEM = (
    "You are a query expansion assistant for a corporate policy RAG system.\n"
    "Given a user question, return exactly 3 alternative phrasings — one per line.\n"
    "No numbering, no bullets, no extra commentary."
)


def node_expand_query(state: AegisState) -> dict:
    """
    Generate 3 alternative phrasings of the user query via LangChain LCEL.
    Stores [original] + variants in state.query_variants.
    Logs all messages (System, Human, AI, Tool) for audit.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    query    = state.query
    prompt   = ChatPromptTemplate.from_messages([
        ("system", _EXPAND_SYSTEM),
        ("human",  "{query}"),
    ])
    chain    = prompt | _llm_creative() | StrOutputParser()
    sys_msg  = SystemMessage(content=_EXPAND_SYSTEM)
    user_msg = HumanMessage(content=query)

    try:
        raw      = chain.invoke({"query": query})
        variants = [v.strip() for v in raw.strip().split("\n") if v.strip()][:3]
        ai_msg   = AIMessage(content=raw)
    except Exception as exc:
        variants = []
        ai_msg   = AIMessage(content=f"expand_query failed: {exc}")

    all_variants = [query] + variants
    log = tool_log(
        tool_name="expand_query",
        reason="Rephrase the user query into multiple semantic variants to improve "
               "recall across different policy document phrasings.",
        inputs={"query": query},
        outputs={"variants": all_variants},
    )
    return {
        "query_variants": all_variants,
        "messages": [sys_msg, user_msg, ai_msg, log],
    }

# ---------------------------------------------------------------------------
# Node 3 — HyDE
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = (
    "You are a corporate policy writer.\n"
    "Write a short, realistic policy clause (2-4 sentences) that would directly\n"
    "answer the following question. Use formal policy language. No preamble."
)


def node_hyde(state: AegisState) -> dict:
    """
    Generate a hypothetical policy-language answer (HyDE) via LangChain LCEL.
    The HyDE document is embedded alongside query variants for retrieval.
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    query    = state.query
    prompt   = ChatPromptTemplate.from_messages([
        ("system", _HYDE_SYSTEM),
        ("human",  "{query}"),
    ])
    chain    = prompt | _llm() | StrOutputParser()
    sys_msg  = SystemMessage(content=_HYDE_SYSTEM)
    user_msg = HumanMessage(content=query)

    try:
        hyde_doc = chain.invoke({"query": query})
        ai_msg   = AIMessage(content=hyde_doc)
    except Exception as exc:
        hyde_doc = query   # fallback: use raw query
        ai_msg   = AIMessage(content=f"HyDE failed: {exc}")

    log = tool_log(
        tool_name="hyde",
        reason="Generate a hypothetical policy clause that structurally resembles "
               "the target document, improving embedding-space alignment for retrieval.",
        inputs={"query": query},
        outputs={"hyde_document": hyde_doc[:300]},
    )
    return {
        "hyde_document": hyde_doc,
        "messages": [sys_msg, user_msg, ai_msg, log],
    }

# ---------------------------------------------------------------------------
# Node 4 — Broad Retrieval (PineconeVectorStore per variant)
# ---------------------------------------------------------------------------

def node_retrieve(state: AegisState) -> dict:
    """
    Hybrid retrieval: dense (Pinecone) + sparse (TF-IDF) fused via RRF.

    Step 1 — Dense retrieval:
        Query Pinecone once per query variant (query_variants + hyde_document).
        Uses LangChain VectorStoreRetriever with optional category pre-filter.
        Deduplicates the pooled results by (document_id, chunk_text[:60]).

    Step 2 — Sparse TF-IDF retrieval:
        Fit a TfidfIndex on the deduplicated dense pool.
        Score every chunk against the original user query.
        This rewards chunks containing exact policy terms: dollar amounts,
        day counts, acronyms, section identifiers.

    Step 3 — Reciprocal Rank Fusion (RRF):
        Merge the dense ranking and the TF-IDF ranking using RRF (k=60).
        RRF is rank-based, so different score magnitudes don't need normalising.
        The fused ranking is stored back in broad_results as the final pool.

    All chunks validated through ChunkResult Pydantic model.
    TF-IDF decisions logged in ToolMessage for audit.
    """
    category  = state.detected_category
    broad_k   = state.top_k                     # lecture: "experiment with topK"
    variants  = state.query_variants or [state.query]
    hyde_doc  = state.hyde_document
    # Trim to num_queries variants (lecture: "experiment with numQueries")
    all_qs    = (variants + ([hyde_doc] if hyde_doc else []))[:state.num_queries + 1]

    search_kwargs: dict = {"k": broad_k}
    if category:
        search_kwargs["filter"] = {"policy_category": {"$eq": category}}

    retriever = _vector_store().as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )

    seen:   set[str] = set()
    pooled: list[ChunkResult] = []
    per_query_counts: dict[str, int] = {}

    # ── Step 1: Dense retrieval ──────────────────────────────────────────
    for q in all_qs:
        try:
            docs = retriever.invoke(q)
        except Exception:
            per_query_counts[q[:40]] = 0
            continue

        count = 0
        for doc in docs:
            key = (doc.metadata.get("document_id", "") +
                   "|" + doc.page_content[:60])
            if key in seen:
                continue
            seen.add(key)
            try:
                chunk = ChunkResult(
                    chunk_id=doc.metadata.get("document_id", "unknown")
                             + f"_{len(pooled)}",
                    document_id=doc.metadata.get("document_id", "unknown"),
                    policy_category=doc.metadata.get("policy_category", "General"),
                    policy_owner=doc.metadata.get("policy_owner", "Unknown"),
                    effective_date=doc.metadata.get("effective_date", ""),
                    h1_header=doc.metadata.get("h1_header", ""),
                    h2_header=doc.metadata.get("h2_header", ""),
                    chunk_text=doc.page_content or doc.metadata.get("chunk_text", ""),
                    is_table=bool(doc.metadata.get("is_table", False)),
                    vector_score=float(doc.metadata.get("score", 0.0)),
                )
                pooled.append(chunk)
                count += 1
            except Exception:
                pass
        per_query_counts[q[:40]] = count

    dense_log = tool_log(
        tool_name="retrieve_dense",
        reason=f"Embed {len(all_qs)} query variants and HyDE doc; query Pinecone "
               f"(Top-{broad_k} each, num_queries={state.num_queries}) with "
               + (f"category pre-filter='{category}'" if category else "no category pre-filter")
               + ". Deduplicate pooled results.",
        inputs={"queries": len(all_qs), "category_filter": category, "broad_k": broad_k, "num_queries": state.num_queries},
        outputs={"total_unique": len(pooled), "per_query": per_query_counts},
    )

    if not pooled:
        return {"broad_results": [], "rrf_applied": False,
                "tfidf_top_indices": [], "messages": [dense_log]}

    # ── Step 2: TF-IDF sparse scoring on the dense pool ─────────────────
    # We index the dense candidates — no separate corpus index needed.
    # Using the original user query (not expansions) for precision focus.
    original_query = state.query
    corpus_texts   = [c.chunk_text for c in pooled]
    tfidf_idx      = TfidfIndex(corpus_texts)
    tfidf_ranked   = tfidf_idx.query(original_query, top_k=len(pooled))

    tfidf_top_indices = [tr.chunk_index for tr in tfidf_ranked[:10]]
    tfidf_top_scores  = {
        pooled[tr.chunk_index].chunk_id: round(tr.tfidf_score, 4)
        for tr in tfidf_ranked[:10]
        if tr.chunk_index < len(pooled)
    }

    tfidf_log = tool_log(
        tool_name="retrieve_tfidf",
        reason="Fit TF-IDF (sublinear_tf, bigrams, token_pattern preserving $/%/.) "
               "on the dense candidate pool. Score every chunk against the raw user "
               "query to surface exact-term matches (dollar amounts, day counts, "
               "acronyms, section IDs) that dense embeddings may underweight.",
        inputs={"corpus_size": len(corpus_texts), "query": original_query[:80]},
        outputs={"top_tfidf_chunk_ids": tfidf_top_scores,
                 "sklearn_available": tfidf_idx.is_ready},
    )

    # ── Step 3: Reciprocal Rank Fusion ───────────────────────────────────
    fused = reciprocal_rank_fusion(
        dense_chunks=pooled,
        tfidf_results=tfidf_ranked,
        k=60,
    )

    rrf_log = tool_log(
        tool_name="retrieve_rrf",
        reason="Merge dense ranking and TF-IDF ranking via Reciprocal Rank Fusion "
               "(k=60). RRF is rank-based — no score normalisation needed across "
               "different score distributions. Chunks appearing highly in both lists "
               "get the highest fused score.",
        inputs={"dense_count": len(pooled), "tfidf_count": len(tfidf_ranked), "k": 60},
        outputs={"fused_count": len(fused),
                 "top3_after_rrf": [
                     {"id": c.chunk_id, "rrf_score": c.vector_score}
                     for c in fused[:3]
                 ]},
    )

    return {
        "broad_results":      fused,
        "tfidf_top_indices":  tfidf_top_indices,
        "rrf_applied":        True,
        "messages":           [dense_log, tfidf_log, rrf_log],
    }

# ---------------------------------------------------------------------------
# Node 5 — Post-Filter (keep most recent version per document)
# ---------------------------------------------------------------------------

def node_post_filter(state: AegisState) -> dict:
    """
    For each document_id in broad_results, keep only chunks with the
    most recent effective_date. This eliminates stale policy versions.
    Pure Python — no LLM call, fully deterministic and auditable.
    """
    matches = state.broad_results
    if not matches:
        log = tool_log("post_filter", "No results to filter.", {}, {"kept": 0})
        return {"broad_results": [], "messages": [log]}

    latest: dict[str, str] = {}
    for c in matches:
        if c.effective_date > latest.get(c.document_id, ""):
            latest[c.document_id] = c.effective_date

    filtered = [
        c for c in matches
        if c.effective_date == latest.get(c.document_id, "")
    ]
    dropped = len(matches) - len(filtered)

    log = tool_log(
        tool_name="post_filter",
        reason="Remove stale policy versions. For each document_id, keep only "
               "chunks where effective_date matches the most recent date seen. "
               "Deterministic — no LLM call.",
        inputs={"total_in": len(matches)},
        outputs={"kept": len(filtered), "dropped": dropped,
                 "latest_dates": latest},
    )
    return {"broad_results": filtered, "messages": [log]}

# ---------------------------------------------------------------------------
# Node 6 — Cross-Encoder Reranking
# ---------------------------------------------------------------------------

def node_rerank(state: AegisState) -> dict:
    """
    Two-stage reranking for maximum precision:

    Stage 1 — Cross-Encoder (ms-marco-MiniLM-L-6-v2):
        Reads query + chunk text simultaneously. Scores semantic relevance
        holistically. Returns Top-FINAL_K from the broad pool.

    Stage 2 — TF-IDF Calibration Bonus:
        After cross-encoder scoring, apply a weighted TF-IDF term-overlap
        bonus (alpha=0.15 default) to nudge exact-term matches upward.

        calibrated_score = cross_encoder_score + alpha × tfidf_score

        Rationale: the cross-encoder already captures semantic relevance
        well. The TF-IDF bonus specifically targets policy queries with
        exact numerical/acronym content where the encoder may assign
        similar scores to two chunks but one contains the exact figure.
        Alpha=0.15 is a light touch — tiebreaker weight, not override.

    Scores are written back into ChunkResult.rerank_score.
    All decisions logged as ToolMessage for audit.
    """
    from langchain_core.documents import Document
    # CRASH FIX: CrossEncoderReranker from langchain_community fails on many
    # Python 3.14 / newer LangChain deployments (screenshot confirmed crash).
    # Use sentence_transformers.CrossEncoder directly — always available when
    # sentence-transformers is installed, no LangChain version coupling.
    try:
        from sentence_transformers import CrossEncoder as _SentCE
        _USE_SENT_CE = True
    except ImportError:
        _USE_SENT_CE = False
        _log.warning("sentence_transformers not available — reranking disabled")

    query  = state.query
    chunks = state.broad_results
    alpha  = state.tfidf_calibration_alpha   # from Pydantic-validated state

    if not chunks:
        log = tool_log("rerank", "No chunks to rerank.", {}, {"top_k": 0})
        return {"reranked_chunks": [], "messages": [log]}

    # A/B test flag — skip cross-encoder when use_reranking=False
    if not state.use_reranking:
        top = chunks[:FINAL_K]
        log = tool_log(
            tool_name="rerank_skipped",
            reason="use_reranking=False — returning top-K by RRF/vector score only. "
                   "Use this config in A/B testing to measure reranker impact.",
            inputs={"use_reranking": False, "top_k": FINAL_K},
            outputs={"selected": [c.chunk_id for c in top]},
        )
        return {"reranked_chunks": top, "messages": [log]}

    lc_docs = [
        Document(page_content=c.chunk_text, metadata=c.model_dump())
        for c in chunks
    ]

    # ── Stage 1: Cross-Encoder (direct sentence_transformers — crash-safe) ──
    ce_error = None
    if _USE_SENT_CE:
        try:
            _ce = _SentCE("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs  = [[query, d.page_content] for d in lc_docs]
            raw_scores = _ce.predict(pairs)
            # Convert to list of floats; handle tensor or ndarray
            if hasattr(raw_scores, "tolist"):
                raw_scores = raw_scores.tolist()
            raw_scores = [float(s) for s in raw_scores]
            # Sort docs by score descending, take top FINAL_K
            ranked_pairs = sorted(
                zip(raw_scores, lc_docs), key=lambda x: x[0], reverse=True
            )[:FINAL_K]
            scores = [s for s, _ in ranked_pairs]
            ranked = [d for _, d in ranked_pairs]
        except Exception as exc:
            _log.error(f"CrossEncoder failed: {exc} — falling back to vector rank")
            ce_error = str(exc)
            ranked   = lc_docs[:FINAL_K]
            scores   = [0.0] * len(ranked)
    else:
        ce_error = "sentence_transformers not installed"
        ranked   = lc_docs[:FINAL_K]
        scores   = [0.0] * len(ranked)

    reranked = _rebuild_chunk_results(ranked, chunks, scores)

    ce_log = tool_log(
        tool_name="rerank_cross_encoder",
        reason=f"Cross-Encoder (ms-marco-MiniLM-L-6-v2) re-scores all {len(chunks)} "
               f"candidates by reading query+chunk simultaneously, then prunes to "
               f"Top-{FINAL_K}. Solves 'Lost in the Middle' problem.",
        inputs={"query": query[:80], "candidates": len(chunks), "top_n": FINAL_K},
        outputs={
            "selected_before_calibration": [
                {"id": r.chunk_id, "ce_score": r.rerank_score, "doc": r.document_id}
                for r in reranked
            ],
            "error": ce_error,
        },
    )

    # ── Stage 2: TF-IDF Calibration Bonus ───────────────────────────────
    calibrated = tfidf_calibrate_scores(query=query, chunks=reranked, alpha=alpha)

    cal_log = tool_log(
        tool_name="rerank_tfidf_calibration",
        reason=f"Apply TF-IDF term-overlap bonus (alpha={alpha}) to cross-encoder "
               "scores. Formula: calibrated = ce_score + alpha × tfidf_score. "
               "Targets precision for exact policy terms: dollar amounts, day counts, "
               "acronyms, section IDs. Alpha is small enough to act as a tiebreaker "
               "without overriding strong semantic matches.",
        inputs={"query": query[:80], "alpha": alpha, "chunks_in": len(reranked)},
        outputs={
            "final_top_k": [
                {"id": c.chunk_id,
                 "calibrated_score": round(c.rerank_score, 4),
                 "h2": c.h2_header}
                for c in calibrated
            ]
        },
    )

    return {"reranked_chunks": calibrated, "messages": [ce_log, cal_log]}


def _rebuild_chunk_results(
    ranked_docs: list,
    original_chunks: list[ChunkResult],
    scores: list[float],
) -> list[ChunkResult]:
    """
    Match ranked LangChain Documents back to original ChunkResult objects,
    inject rerank_score, and re-validate through Pydantic.
    """
    result: list[ChunkResult] = []
    for doc, score in zip(ranked_docs, scores):
        # Find the original ChunkResult by chunk_text prefix match
        original = next(
            (c for c in original_chunks
             if c.chunk_text[:60] == doc.page_content[:60]),
            None,
        )
        if original:
            data = original.model_dump()
            data["rerank_score"] = score
            try:
                result.append(ChunkResult(**data))
            except Exception:
                pass
        else:
            # Build from doc metadata (fallback)
            meta = doc.metadata
            try:
                result.append(ChunkResult(
                    chunk_id=meta.get("chunk_id", "unknown"),
                    document_id=meta.get("document_id", "unknown"),
                    policy_category=meta.get("policy_category", "General"),
                    policy_owner=meta.get("policy_owner", "Unknown"),
                    effective_date=meta.get("effective_date", ""),
                    h1_header=meta.get("h1_header", ""),
                    h2_header=meta.get("h2_header", ""),
                    chunk_text=doc.page_content,
                    is_table=bool(meta.get("is_table", False)),
                    vector_score=float(meta.get("vector_score", 0.0)),
                    rerank_score=score,
                ))
            except Exception:
                pass
    return result

# ---------------------------------------------------------------------------
# Node 7 — Answer Generation (create_stuff_documents_chain)
# ---------------------------------------------------------------------------

# ── STRUCTURED SYSTEM PROMPT — anti-hallucination enforced ──────────────────
# Sameer lecture: "No prompt control → hallucination risk"
# Fix: explicit SYSTEM constraint + required fallback phrase
_GEN_SYSTEM = (
    "You are an authoritative corporate policy assistant.\n"
    "STRICT RULES — follow ALL of these:\n"
    "  1. Answer ONLY from the provided policy context below.\n"
    "  2. NEVER invent, infer, or extrapolate facts not present in the context.\n"
    "  3. NEVER cite documents, sections, or policies not mentioned in context.\n"
    "  4. If the answer is not in the context, respond EXACTLY with:\n"
    '     \"Not found in context. Please check that the relevant policy document has been ingested.\"\n'
    "  5. When answering, cite the section name and document ID where the fact appears.\n"
    "  6. For numeric facts (dollar amounts, day counts, percentages), quote them exactly.\n\n"
    "Context:\n{context}"
)


def node_generate(state: AegisState) -> dict:
    """
    Build the final grounded answer via LangChain create_stuff_documents_chain.
    Logs the full prompt context summary and answer as messages.
    """
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.documents import Document

    query  = state.query
    chunks = state.reranked_chunks

    if not chunks:
        answer = (
            "I could not find relevant policy information to answer your question. "
            "Please try rephrasing or check that the relevant policy document has been ingested."
        )
        log = tool_log(
            tool_name="generate_answer",
            reason="No reranked chunks available — returning safe fallback answer.",
            inputs={"query": query},
            outputs={"answer": answer},
        )
        return {"final_answer": answer, "messages": [log]}

    # ── Token limit enforcement (lecture: "enforce token limits") ─────────
    # Trim chunk texts so total context stays within max_context_tokens budget.
    # Approximate: 1 token ≈ 4 chars.
    max_chars   = state.max_context_tokens * 4
    used_chars  = 0
    kept_chunks = []
    for c in chunks:
        if used_chars + len(c.chunk_text) > max_chars:
            # Truncate last chunk to fit budget
            remaining = max_chars - used_chars
            if remaining > 200:
                from graph_state import ChunkResult as CR
                data = c.model_dump()
                data["chunk_text"] = c.chunk_text[:remaining] + "…"
                try:
                    kept_chunks.append(CR(**data))
                except Exception:
                    pass
            break
        kept_chunks.append(c)
        used_chars += len(c.chunk_text)

    token_log = tool_log(
        tool_name="context_token_limit",
        reason=f"Enforce max_context_tokens={state.max_context_tokens} "
               f"(~{max_chars} chars). Prevents 'Lost in the Middle' and "
               "controls LLM cost. Lecture: 'enforce token limits'.",
        inputs={"max_tokens": state.max_context_tokens, "chunks_in": len(chunks)},
        outputs={"chunks_kept": len(kept_chunks), "chars_used": used_chars},
    )

    # ── Optional cheap-model summarisation (lecture: "use cheaper/smaller
    #    models for summarisation to reduce token costs") ─────────────────
    if state.use_summarisation and kept_chunks:
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate as CPT
        from langchain_core.output_parsers import StrOutputParser
        cheap_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0,
                               api_key=os.getenv("OPENAI_API_KEY", ""))
        raw_ctx   = "\n\n---\n\n".join(c.chunk_text for c in kept_chunks)
        sum_prompt = CPT.from_messages([
            ("system",
             "Summarise these policy excerpts in ≤300 tokens. "
             "Preserve all numbers, dates, and dollar amounts."),
            ("human", f"Question: {query}\n\nExcerpts:\n{raw_ctx}"),
        ])
        try:
            summarised_ctx = (sum_prompt | cheap_llm | StrOutputParser()).invoke({})
        except Exception:
            summarised_ctx = raw_ctx   # fallback: use full context
        sum_log = tool_log(
            tool_name="summarise_context",
            reason="use_summarisation=True — compress context with gpt-3.5-turbo "
                   "to reduce input tokens for the final answer LLM. "
                   "Lecture: 'use cheaper/smaller models for summarisation'.",
            inputs={"chunks": len(kept_chunks), "query": query[:80]},
            outputs={"summary_chars": len(summarised_ctx)},
        )
        # Wrap summary as single Document
        from langchain_core.documents import Document as LCDoc
        lc_docs = [LCDoc(page_content=summarised_ctx,
                         metadata={"source": "summarised_context"})]
        extra_messages = [token_log, sum_log]
    else:
        lc_docs = [
            Document(
                page_content=c.chunk_text,
                metadata={
                    "source": f"{c.h1_header} / {c.h2_header}".strip(" /"),
                    "document_id":     c.document_id,
                    "policy_category": c.policy_category,
                    "effective_date":  c.effective_date,
                    "rerank_score":    c.rerank_score,
                },
            )
            for c in kept_chunks
        ]
        extra_messages = [token_log]

    prompt = ChatPromptTemplate.from_messages([
        ("system", _GEN_SYSTEM),
        ("human",  "{input}"),
    ])

    sys_msg  = SystemMessage(content=_GEN_SYSTEM.split("{context}")[0].strip())
    user_msg = HumanMessage(content=query)

    try:
        chain  = create_stuff_documents_chain(llm=_llm(), prompt=prompt)
        answer = chain.invoke({"input": query, "context": lc_docs})
        ai_msg = AIMessage(content=answer)
    except Exception as exc:
        answer = f"Answer generation failed: {exc}"
        ai_msg = AIMessage(content=answer)

    context_summary = [
        {"doc": c.document_id, "h2": c.h2_header, "score": round(c.rerank_score, 4)}
        for c in chunks
    ]

    # ── Anti-hallucination check ──────────────────────────────────────────
    # Verifies the answer doesn't contain terms absent from the context.
    # If hallucination risk is detected, answer is replaced with safe fallback.
    context_texts = [c.chunk_text for c in kept_chunks]
    is_risky, halluc_score = check_hallucination_risk(answer, context_texts)
    if is_risky:
        fb = handle_fallback(
            FallbackReason.HALLUCINATION, "node_generate",
            query=query,
            detail=f"Hallucination risk score={halluc_score:.2%}. "
                   f"Answer replaced with safe fallback.",
            halluc_score=halluc_score,
        )
        answer = fb.answer
        ai_msg = AIMessage(content=answer)
        extra_messages.append(fb.to_tool_message())

    halluc_log = tool_log(
        tool_name="hallucination_check",
        reason="Post-generation hallucination check: verify answer tokens appear "
               "in retrieved context vocabulary. Replace with fallback if risk detected.",
        inputs={"query": query[:80], "answer_chars": len(answer),
                "context_chunks": len(context_texts)},
        outputs={"is_risky": is_risky, "halluc_score": halluc_score,
                 "replaced": is_risky},
    )

    log = tool_log(
        tool_name="generate_answer",
        reason=f"Pass Top-{len(chunks)} reranked policy chunks to LLM via "
               "create_stuff_documents_chain. LLM must cite only the provided context.",
        inputs={"query": query, "chunk_count": len(chunks)},
        outputs={"answer_chars": len(answer), "context_used": context_summary,
                 "halluc_score": halluc_score, "is_risky": is_risky},
    )
    return {
        "final_answer": answer,
        "hallucination_risk": is_risky,
        "messages": extra_messages + [sys_msg, user_msg, ai_msg, halluc_log, log],
    }
