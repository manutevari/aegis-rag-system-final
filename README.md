# Decision-Grade Corporate Policy RAG Chatbot

A production-grade, context-aware RAG chatbot for navigating complex, interconnected, and highly numerical corporate policy documents. The implementation follows the Project Aegis guidelines for semantic ingestion, metadata-aware retrieval, query transformation, reranking, and grounded answer verification.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env .env.local

# Local-first mode, no hosted model keys required
RAG_EMBEDDINGS_PROVIDER=hash VECTOR_BACKEND=chroma LLM_PROVIDER=local_auto python policy_ingestion.py
RAG_EMBEDDINGS_PROVIDER=hash VECTOR_BACKEND=chroma LLM_PROVIDER=local_auto uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Local LLM Orchestration

AEGIS now routes generation through a local-only decision manager in `app/core/llm_decision_manager.py`. It skips hosted models and tries only these local runtimes:

1. `ollama`, if the Ollama server is reachable and the requested model is available.
2. `llama_cpp`, using a local OpenAI-compatible llama.cpp server.
3. `mistral_local`, using a local OpenAI-compatible Mistral endpoint.
4. `extractive`, the deterministic fallback when no local runtime is working.

Configure the local runtime stack with environment variables or the Streamlit sidebar:

```bash
LLM_PROVIDER=local_auto
LOCAL_LLM_ORDER=ollama,llama_cpp,mistral_local
LOCAL_ORCHESTRATION_MODEL=llama3.1
LOCAL_GENERATION_MODEL=mistral
LOCAL_SUMMARY_MODEL=mistral

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral

LLAMA_CPP_BASE_URL=http://localhost:8080/v1
LLAMA_CPP_MODEL=local-model

MISTRAL_LOCAL_BASE_URL=http://localhost:8000/v1
MISTRAL_LOCAL_MODEL=mistral
```

You can force one runtime by setting `LLM_PROVIDER=ollama`, `LLM_PROVIDER=llama_cpp`, `LLM_PROVIDER=mistral_local`, or `LLM_PROVIDER=extractive`.

The decision manager is node-aware: orchestration-style nodes use `LOCAL_ORCHESTRATION_MODEL`, summarization uses `LOCAL_SUMMARY_MODEL`, and grounded answer generation uses `LOCAL_GENERATION_MODEL` unless a node passes an explicit model override.

## Retrieval Stack

The default retrieval stack is local-friendly:

- Hash embeddings for deterministic offline retrieval.
- Chroma for local vector storage.
- Lexical/local reranking by default.
- Pinecone, OpenAI, Gemini, and Cohere settings remain in the code for compatibility, but the local LLM decision manager does not select hosted models.

Build and run locally:

```bash
python policy_ingestion.py
uvicorn api:app --reload --host 0.0.0.0 --port 8000
streamlit run streamlit_app.py --server.fileWatcherType none
```

## Pinecone Index

If you explicitly enable Pinecone with hosted embeddings, create a dense Pinecone index matching the configured embedding dimension and metric. Local default mode does not require Pinecone.

## AEGIS Ingestion Engine

`policy_ingestion.py` implements the guideline ingestion pipeline:

1. Markdown-aware semantic chunking parses `#`, `##`, and `###` headers so each chunk carries its section path.
2. Table preservation keeps Markdown tables intact; oversized tables are chunked by rows with the header repeated.
3. Sequential overlap adds a 10-15% context bridge between neighboring chunks.
4. Metadata extraction attaches `document_id`, `policy_category`, `policy_owner`, `effective_date`, `last_revised`, `h1_header`, `h2_header`, `h3_header`, `source_path`, and `section_path` to every chunk.
5. Ingestion verification blocks indexing if required metadata is missing or a chunk is empty.
6. Upsertion batches the verified chunks into the configured vector store.

## Advanced Retrieval Pipeline

`app/nodes/retrieval.py` implements the guideline retrieval pipeline:

1. Query expansion generates policy-oriented rewrites such as rideshare, cab fare, and ground transportation variants for a taxi question.
2. HyDE-style search text creates a hypothetical policy answer and retrieves against that richer text.
3. Metadata pre-filtering applies category filters such as `policy_category == travel` before vector search.
4. Broad retrieval pools up to 25 chunks across the raw query, expanded queries, and HyDE query.
5. Latest-version post-filtering keeps the newest effective policy version when multiple versions of a policy family are retrieved.
6. Reranking passes only the top 5 into generation, with lexical fallback if no hosted reranker is configured.

## Architecture

```text
User query
  -> planner / router
  -> advanced retrieval
       -> query expansion
       -> HyDE-style search text
       -> metadata pre-filter
       -> local embeddings + vector retrieval
       -> latest-version post-filter
       -> rerank top 5
  -> context assembler / token manager
  -> local LLM decision manager
       -> Ollama if working
       -> llama.cpp if working
       -> Mistral local if working
       -> extractive fallback
  -> grounded generator
  -> confidence and verifier
  -> retry or HITL fallback
  -> trace end
```

## Key Files

| File | Purpose |
|------|---------|
| `app/core/llm_decision_manager.py` | Local-only node-aware model orchestration for Ollama, llama.cpp, and Mistral local |
| `app/core/settings.py` | Pydantic runtime settings for local runtimes, Chroma, Pinecone compatibility, and fallbacks |
| `policy_ingestion.py` | Project Aegis ingestion engine: markdown chunking, table preservation, metadata extraction, verification, upsert |
| `app/core/vector_store.py` | Embeddings, vector store, Chroma/hash fallbacks, metadata-filtered retrieval |
| `app/nodes/retrieval.py` | Query expansion, HyDE-style retrieval, metadata filters, post-filtering, reranking |
| `app/core/models.py` | Local decision manager entrypoint with extractive fallback |
| `app/graph/workflow.py` | Full LangGraph workflow |
| `app/nodes/planner.py` | Rule-based grade-aware router |
| `app/nodes/verifier.py` | Blocking quality gate |
| `streamlit_app.py` | Primary Streamlit chat UI, local runtime controls, and execution trace viewer |
| `api.py` | FastAPI REST API |
| `tests/test_all.py` | Core unit tests |
| `tests/test_vector_wiring.py` | Ingestion/retrieval wiring regression tests |
| `tests/test_llm_decision_manager.py` | Local LLM orchestration tests |

## Run Tests

```bash
pytest tests/ -v
```

## Docker

```bash
docker-compose up -d
# UI: http://localhost:8501 | API: http://localhost:8000/docs
```

## Add Your Own Policies

Drop `.txt`, `.md`, or `.pdf` files anywhere under `data/`. Streamlit auto-indexes them into the configured vector store on startup, and `python policy_ingestion.py` can rebuild the index manually.
