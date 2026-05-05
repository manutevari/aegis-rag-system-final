from __future__ import annotations

import os
import re
import types
from decimal import Decimal
from typing import List, Optional, Tuple


def apply(app) -> None:
    numeric_re = re.compile(
        r"(?P<prefix>(?:\$|Rs\.?|INR)?)\s*(?P<number>\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?P<suffix>\s*(?:%|percent|days?|hours?|weeks?|months?|years?|usd|dollars?|inr|rs\.?|km|miles?|per\s+day|daily|per\s+diem)?)",
        re.IGNORECASE,
    )

    def graph_node_id(doc) -> str:
        return f"{doc.metadata.get('source', 'Unknown')}::{doc.metadata.get('chunk_index', 0)}"

    def build_policy_graph(chunks: List) -> dict:
        nodes = []
        for doc in chunks:
            meta = doc.metadata
            nodes.append({
                "id": graph_node_id(doc),
                "source": meta.get("source", "Unknown"),
                "section": meta.get("section_header") or meta.get("h1_header") or "Document",
                "category": meta.get("policy_category", "General"),
                "effective_date": meta.get("effective_date", ""),
                "clause_preview": " ".join(doc.page_content.split())[:220],
            })

        by_section = {str(node["section"]).lower(): node["id"] for node in nodes if node.get("section")}
        edges, seen = [], set()
        ref_re = re.compile(
            r"\b(?:see|refer(?:s|red)? to|under|according to|as defined in)\s+"
            r"(?:section|clause|policy)?\s*([A-Za-z0-9][A-Za-z0-9 ._\-/]{1,60})",
            re.IGNORECASE,
        )
        for doc in chunks:
            source_id = graph_node_id(doc)
            for match in ref_re.finditer(doc.page_content):
                ref = match.group(1).strip(" .,:;").lower()
                target_id = next((node_id for section, node_id in by_section.items() if ref and ref in section), "")
                if target_id and target_id != source_id:
                    key = (source_id, target_id, ref)
                    if key not in seen:
                        edges.append({"from": source_id, "to": target_id, "type": "references", "reference": ref})
                        seen.add(key)
        return {"nodes": nodes, "edges": edges}

    def keyword_search(query: str, docs: List, k: int):
        q = app.content_tokens(query)
        if not q:
            return [], []
        ranked = []
        phrase = query.lower().strip()
        for doc in docs:
            text = app.rerank_text(doc).lower()
            d = app.content_tokens(text)
            score = len(q & d) / max(1, len(q))
            if phrase and phrase in text:
                score += 0.25
            if score > 0:
                ranked.append((doc, round(score, 4)))
        ranked.sort(key=lambda item: item[1], reverse=True)
        kept = ranked[:k]
        trace = [
            {
                "rank": rank,
                "score": score,
                "source": doc.metadata.get("source"),
                "section": doc.metadata.get("section_header"),
            }
            for rank, (doc, score) in enumerate(kept, start=1)
        ]
        return [doc for doc, _ in kept], trace

    def decimal_from_text(value: str) -> Optional[Decimal]:
        try:
            return Decimal(str(value).replace(",", "").strip())
        except Exception:
            return None

    def effective_date(doc) -> str:
        date = str(doc.metadata.get("effective_date", ""))
        return date if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else "0000-00-00"

    def numeric_unit(prefix: str, suffix: str) -> str:
        text = f"{prefix} {suffix}".strip().lower()
        if "$" in text or "usd" in text or "dollar" in text or "inr" in text or "rs" in text:
            return "currency"
        if "%" in text or "percent" in text:
            return "percent"
        if any(w in text for w in ["day", "hour", "week", "month", "year"]):
            return "time"
        if any(w in text for w in ["km", "mile"]):
            return "distance"
        return "number"

    def sentence_windows(text: str) -> List[str]:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text or "") if part.strip()]

    def extract_numeric_facts(docs: List) -> List[dict]:
        facts = []
        for doc in docs:
            for sentence in sentence_windows(doc.page_content):
                for match in numeric_re.finditer(sentence):
                    value = decimal_from_text(match.group("number"))
                    if value is None:
                        continue
                    facts.append({
                        "source": doc.metadata.get("source", "Unknown"),
                        "section": doc.metadata.get("section_header") or doc.metadata.get("h1_header") or "Document",
                        "effective_date": effective_date(doc),
                        "category": doc.metadata.get("policy_category", "General"),
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                        "value": str(value),
                        "display_value": match.group(0).strip(),
                        "unit": numeric_unit(match.group("prefix"), match.group("suffix")),
                        "text": sentence[:500],
                    })
        return facts

    def stricter_direction(query: str, facts: List[dict]) -> str:
        text = f"{query} {' '.join(f.get('text', '') for f in facts[:5])}".lower()
        if any(word in text for word in ["minimum", "at least", "no less", "floor", "required"]):
            return "higher"
        return "lower"

    def resolve_numeric_conflict(query: str, facts: List[dict]) -> dict:
        if not facts:
            return {"multiple_values": False, "rule": "none", "selected": None, "candidates": []}
        latest_date = max(f.get("effective_date", "0000-00-00") for f in facts)
        latest = [f for f in facts if f.get("effective_date", "0000-00-00") == latest_date]
        numeric_latest = [(f, decimal_from_text(f["value"])) for f in latest]
        numeric_latest = [(f, v) for f, v in numeric_latest if v is not None]
        selected, rule = latest[0], "latest policy"
        if len(numeric_latest) > 1:
            direction = stricter_direction(query, latest)
            selected, _value = (max if direction == "higher" else min)(numeric_latest, key=lambda item: item[1])
            rule = f"latest policy, then stricter constraint ({direction})"
        values = {f"{f.get('display_value')}|{f.get('effective_date')}|{f.get('source')}" for f in facts}
        return {"multiple_values": len(values) > 1, "rule": rule, "selected": selected, "candidates": facts[:12]}

    def compute_policy_math(query: str, conflict: dict) -> Optional[dict]:
        selected = conflict.get("selected") or {}
        value = decimal_from_text(selected.get("value", ""))
        if value is None:
            return app.calculation(query)
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s+of", query, re.IGNORECASE)
        if pct:
            percent = Decimal(pct.group(1))
            result = (percent / Decimal("100")) * value
            return {
                "expression": f"{percent}% of {selected.get('display_value')}",
                "result": format(result.normalize(), "f"),
                "verified": True,
                "source": selected.get("source"),
                "rule": conflict.get("rule"),
            }
        days = re.search(r"\b(\d+(?:\.\d+)?)\s+days?\b", query, re.IGNORECASE)
        if days and re.search(r"per\s+day|daily|per\s+diem", selected.get("text", ""), re.IGNORECASE):
            count = Decimal(days.group(1))
            result = count * value
            return {
                "expression": f"{count} days * {selected.get('display_value')}",
                "result": format(result.normalize(), "f"),
                "verified": True,
                "source": selected.get("source"),
                "rule": conflict.get("rule"),
            }
        return app.calculation(query)

    def numeric_reasoning(query: str, docs: List) -> dict:
        facts = extract_numeric_facts(docs)
        conflict = resolve_numeric_conflict(query, facts)
        calc = compute_policy_math(query, conflict)
        detected = bool(facts or calc or re.search(r"\d|allowance|limit|maximum|minimum|cap|per diem|reimburse", query, re.IGNORECASE))
        return {
            "pipeline": ["retrieval", "detect numeric logic", "extract variables", "compute", "verify"],
            "detected": detected,
            "variables": facts[:12],
            "calculation": calc,
            "verification": {
                "status": "verified" if (calc or conflict.get("selected")) else "not_applicable",
                "basis": "retrieved policy chunks and Decimal arithmetic",
            },
            "conflict_resolution": conflict,
        }

    def citation_trace(docs: List) -> List[dict]:
        return [
            {
                "citation": f"{doc.metadata.get('source', 'Unknown')} | {doc.metadata.get('section_header') or doc.metadata.get('h1_header') or 'Document'}",
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section_header") or doc.metadata.get("h1_header") or "Document",
                "effective_date": doc.metadata.get("effective_date", ""),
                "category": doc.metadata.get("policy_category", "General"),
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc in docs
        ]

    def resolved_numeric_context(numeric_trace: dict) -> str:
        selected = (numeric_trace.get("conflict_resolution") or {}).get("selected")
        calc = numeric_trace.get("calculation")
        parts = []
        if selected:
            parts.append(
                "[Resolved numeric policy value]\n"
                f"Rule: {numeric_trace['conflict_resolution'].get('rule')}\n"
                f"Value: {selected.get('display_value')}\n"
                f"Source: {selected.get('source')} | Section: {selected.get('section')} | Effective: {selected.get('effective_date')}\n"
                f"Clause: {selected.get('text')}"
            )
        if calc:
            parts.append(
                "[Verified numeric calculation]\n"
                f"Expression: {calc.get('expression')}\n"
                f"Result: {calc.get('result')}\n"
                f"Source: {calc.get('source', 'query/context')}\n"
                f"Rule: {calc.get('rule', 'Decimal arithmetic')}"
            )
        return "\n\n".join(parts)

    def enhanced_build_index(upload_payload, repo_fingerprint, use_openai_embeddings, metadata_key_hash, metadata_key):
        del repo_fingerprint, metadata_key_hash
        documents = app.load_documents(upload_payload, metadata_key=metadata_key)
        if not documents:
            return None
        chunks = app.split_documents(documents)
        if not chunks:
            return None

        embeddings = app.HashEmbeddings()
        backend = "Local hash"
        warning = ""
        allow_hosted = os.getenv("AEGIS_ALLOW_HOSTED_EMBEDDINGS", "").lower() in {"1", "true", "yes"}
        if use_openai_embeddings and allow_hosted and app.OpenAIEmbeddings and app.provider_key("OpenAI"):
            try:
                embeddings = app.OpenAIEmbeddings(api_key=app.provider_key("OpenAI"))
                backend = "OpenAI"
            except Exception as exc:
                warning = f"OpenAI embeddings failed before indexing, so local embeddings were used: {exc}"

        try:
            main_index = app.FAISS.from_documents(chunks, embeddings)
        except Exception as exc:
            embeddings = app.HashEmbeddings()
            backend = "Local hash fallback"
            warning = f"Hosted embeddings failed, so local embeddings were used: {exc}"
            main_index = app.FAISS.from_documents(chunks, embeddings)

        category_indexes, category_chunks = {}, {}
        for category in sorted({chunk.metadata.get("policy_category", "General") for chunk in chunks}):
            cat_chunks = [chunk for chunk in chunks if chunk.metadata.get("policy_category", "General") == category]
            category_chunks[category] = cat_chunks
            category_indexes[category] = app.FAISS.from_documents(cat_chunks, embeddings)

        return types.SimpleNamespace(
            vectorstore=main_index,
            category_indexes=category_indexes,
            sources=sorted({doc.metadata.get("source", "Unknown") for doc in documents}),
            metadata_rows=app.metadata_rows(documents),
            chunk_count=len(chunks),
            embedding_backend=backend,
            embedding_warning=warning,
            chunks=chunks,
            category_chunks=category_chunks,
            policy_graph=build_policy_graph(chunks),
        )

    def enhanced_retrieve(index, query: str, llm, reranker_provider: str, cohere_key: str, cohere_model: str) -> Tuple[str, List[str], dict]:
        pack = app.query_pack(query, llm)
        graph = getattr(index, "policy_graph", {"nodes": [], "edges": []}) or {"nodes": [], "edges": []}
        route = app.choose_category(query, index.category_indexes.keys())
        if route["applied"] and route["category"] not in index.category_indexes:
            trace = {
                "query_pack": pack,
                "metadata_prefilter": route,
                "date_post_filter": {},
                "hybrid_retrieval": [],
                "unique_chunks": 0,
                "reranker": {"provider": "skipped", "input_chunks": 0, "output_chunks": 0},
                "context_chunks": 0,
                "numeric_reasoning": numeric_reasoning(query, []),
                "conflict_resolution": {"multiple_values": False, "rule": "metadata prefilter excluded all chunks", "selected": None},
                "citations": [],
                "policy_graph": {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "sample_edges": graph["edges"][:8]},
            }
            return "", [], trace

        vectorstore = index.category_indexes.get(route["category"], index.vectorstore) if route["applied"] else index.vectorstore
        scope_docs = index.category_chunks.get(route["category"], index.chunks) if route["applied"] else index.chunks
        seen, pooled, runs = set(), [], []
        searches = [("raw", pack["raw"])]
        searches += [(f"expansion_{i + 1}", q) for i, q in enumerate(pack["expansions"])]
        searches += [("hyde", pack["hyde"])]
        for kind, search_text in searches:
            vector_docs = vectorstore.similarity_search(search_text, k=app.BROAD_RETRIEVAL_K)
            keyword_docs, keyword_trace = keyword_search(search_text, scope_docs, app.BROAD_RETRIEVAL_K)
            combined = [*vector_docs, *keyword_docs]
            runs.append({
                "type": kind,
                "vector_retrieved": len(vector_docs),
                "keyword_retrieved": len(keyword_docs),
                "sources": sorted({d.metadata.get("source") for d in combined}),
                "top_keyword_scores": keyword_trace[:5],
            })
            for doc in combined:
                key = app.doc_key(doc)
                if key not in seen:
                    pooled.append(doc)
                    seen.add(key)

        latest_docs, date_trace = app.keep_latest(pooled)
        broad = latest_docs[:app.BROAD_RETRIEVAL_K]
        final_docs, rerank_trace = app.rerank(query, broad, reranker_provider, cohere_key, cohere_model)
        numeric_trace = numeric_reasoning(query, final_docs)
        context_parts = [f"{app.header(doc.metadata)}\n{doc.page_content}" for doc in final_docs]
        numeric_context = resolved_numeric_context(numeric_trace)
        if numeric_context:
            context_parts.append(numeric_context)
        context = "\n\n".join(context_parts)
        sources = sorted({doc.metadata.get("source", "Unknown") for doc in final_docs})
        trace = {
            "query_pack": pack,
            "metadata_prefilter": {**route, "matched_chunks": len(pooled)},
            "date_post_filter": date_trace,
            "hybrid_retrieval": runs,
            "broad_k": app.BROAD_RETRIEVAL_K,
            "unique_chunks": len(pooled),
            "rerank_candidates": len(broad),
            "reranker": rerank_trace,
            "context_chunks": len(final_docs),
            "numeric_reasoning": numeric_trace,
            "conflict_resolution": numeric_trace["conflict_resolution"],
            "citations": citation_trace(final_docs),
            "policy_graph": {"nodes": len(graph["nodes"]), "edges": len(graph["edges"]), "sample_edges": graph["edges"][:8]},
            "runs": runs,
        }
        return context, sources, trace

    app.build_index = app.st.cache_resource(show_spinner=False)(enhanced_build_index)
    app.retrieve = enhanced_retrieve
