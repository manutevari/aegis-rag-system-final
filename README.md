# 🛡️ AEGIS Enterprise RAG — LangChain + LangGraph Edition

**Project-compliant** · Context-aware retrieval · Cross-encoder reranking · Full audit trail

---

## Architecture

```
User Query (string)
       │
       ▼  graph.py — StateGraph(AegisState)
  ┌─────────────────────────────────────────────────────────┐
  │  START → input → router → expand_query → hyde           │
  │         → retrieve → post_filter → rerank → generate    │
  │         → END                                           │
  └─────────────────────────────────────────────────────────┘
       │
       ▼
  AegisState.final_answer  +  AegisState.messages (full audit)
```

## File map

| File | Role |
|---|---|
| `graph_state.py` | Pydantic `AegisState` + `ChunkResult` + `tool_log()` helper |
| `graph.py` | LangGraph `StateGraph` — wires nodes, exposes `run_query()` |
| `retrieval_nodes.py` | 7 LangGraph nodes (router → generate) |
| `ingestion.py` | LangChain chunking + embedding + upsert |
| `retrieval.py` | Backwards-compatible shim delegating to graph |
| `streamlit_app.py` | UI with audit trail expander per response |
| `utils.py` | `clean_text`, `truncate`, `format_date`, `load_txt_file` |
| `metadata.py` | Legacy metadata helpers (kept for compat) |

---

## LangGraph nodes

| # | Node | Tool logged | LLM? |
|---|---|---|---|
| 1 | `node_router` | `router` | Keyword-first; LLM fallback only |
| 2 | `node_expand_query` | `expand_query` | ✓ (3 variants) |
| 3 | `node_hyde` | `hyde` | ✓ (hypothetical clause) |
| 4 | `node_retrieve` | `retrieve` | ✗ (embedding only) |
| 5 | `node_post_filter` | `post_filter` | ✗ (pure Python) |
| 6 | `node_rerank` | `rerank` | ✗ (CrossEncoder) |
| 7 | `node_generate` | `generate_answer` | ✓ (final answer) |

---

## Message protocol

Every node appends to `AegisState.messages`:

| Type | When used |
|---|---|
| `SystemMessage` | Prompt / pipeline description |
| `HumanMessage` | User query forwarded to LLM |
| `AIMessage` | Raw LLM text output |
| `ToolMessage` | Audit entry: tool name, **reason called**, inputs, outputs |

`ToolMessage` payload (JSON):
```json
{
  "tool":    "expand_query",
  "reason":  "Rephrase the user query into multiple semantic variants...",
  "inputs":  {"query": "Can I expense a taxi?"},
  "outputs": {"variants": ["Can I expense a taxi?", "Policy on rideshares...", ...]}
}
```

---

## Pydantic enforcement

- `AegisState` — validates all fields on every state update; `@model_validator` ensures `reranked_chunks ≤ broad_results`
- `ChunkResult` — validates `policy_category` against `VALID_CATEGORIES`; normalises `effective_date` to `YYYY-MM-DD` or empty
- `ChunkMetadata` (ingestion) — rejects blank `chunk_text`; coerces invalid categories to `"General"`
- `StructuredOutputParser` — typed metadata extraction with `ResponseSchema`

---

## Router logic

```
query.lower()  ──► scan KEYWORD_MAP (O(1), ~40 entries)
                        │
              match?  ──┼──► return category  (confidence="high", 0 LLM calls)
                        │
              no match ─┴──► LLM classifier   (confidence="low", 1 LLM call)
```

The keyword map is in `retrieval_nodes.KEYWORD_MAP` — edit freely without touching graph logic.

---

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY, PINECONE_API_KEY
streamlit run streamlit_app.py
```

## Quick test

```python
from graph import run_query
result = run_query("What is the per diem rate for international travel?")
print(result["answer"])
# Audit trail
for msg in result["messages"]:
    print(type(msg).__name__, msg.content[:80])
```
