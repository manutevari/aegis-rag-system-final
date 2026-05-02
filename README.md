# Decision-Grade Corporate Policy RAG Chatbot

A production-grade, context-aware RAG chatbot for navigating complex, interconnected, and highly numerical corporate policy documents. The implementation follows the Project Aegis guidelines for semantic ingestion, metadata-aware retrieval, query transformation, reranking, and grounded answer verification.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env .env.local

# Build the shared policy index
RAG_EMBEDDINGS_PROVIDER=hash python policy_ingestion.py

# Streamlit UI -> http://localhost:8501
RAG_EMBEDDINGS_PROVIDER=hash LLM_PROVIDER=extractive streamlit run streamlit_app.py --server.fileWatcherType none

# REST API -> http://localhost:8000/docs
RAG_EMBEDDINGS_PROVIDER=hash LLM_PROVIDER=extractive uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Local Runtime

The default runtime is deliberately local and deployment-safe:

- `LLM_PROVIDER=extractive` avoids a hard dependency on Ollama and prevents `localhost:11434` connection errors.
- `RAG_EMBEDDINGS_PROVIDER=hash` avoids `torch`, `torchvision`, and `transformers` imports in constrained Streamlit/FastAPI environments.
- Chroma remains the shared vector store and supports metadata filters for policy category, document id, effective date, source, and section headers.

Optional dense-model mode is still available when the environment supports it:

```bash
# Prefer cached Hugging Face embeddings, then fallback to hash
RAG_EMBEDDINGS_PROVIDER=local python policy_ingestion.py

# Optional Ollama generation
ollama run llama3
LLM_PROVIDER=ollama uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Optional cross-encoder reranking when sentence-transformers is installed
RERANK_PROVIDER=cross_encoder RERANK_MODEL=BAAI/bge-reranker-base uvicorn api:app --reload
```

## AEGIS Ingestion Engine

`policy_ingestion.py` implements the guideline ingestion pipeline:

1. Markdown-aware semantic chunking parses `#`, `##`, and `###` headers so each chunk carries its section path.
2. Table preservation keeps Markdown tables intact; oversized tables are chunked by rows with the header repeated.
3. Sequential overlap adds a 10-15% context bridge between neighboring chunks.
4. Metadata extraction attaches `document_id`, `policy_category`, `policy_owner`, `effective_date`, `last_revised`, `h1_header`, `h2_header`, `h3_header`, `source_path`, and `section_path` to every chunk.
5. Ingestion verification blocks indexing if required metadata is missing or a chunk is empty.
6. Upsertion batches the verified chunks into the shared Chroma collection with their metadata payloads.

## Advanced Retrieval Pipeline

`app/nodes/retrieval.py` implements the guideline retrieval pipeline:

1. Query expansion generates policy-oriented rewrites such as rideshare, cab fare, and ground transportation variants for a taxi question.
2. HyDE-style search text creates a hypothetical policy answer and retrieves against that richer text.
3. Metadata pre-filtering applies category filters such as `policy_category == travel` before vector search.
4. Broad retrieval pools up to 25 chunks across the raw query, expanded queries, and HyDE query.
5. Latest-version post-filtering keeps the newest effective policy version when multiple versions of a policy family are retrieved.
6. Reranking scores pooled chunks and passes only the top 5 into generation to reduce lost-in-the-middle failures.

## Architecture

```text
User query
  -> planner / router
  -> advanced retrieval
       -> query expansion
       -> HyDE-style search text
       -> metadata pre-filter
       -> broad vector retrieval
       -> latest-version post-filter
       -> rerank top 5
  -> context assembler / token manager
  -> grounded generator
  -> confidence and verifier
  -> retry or HITL fallback
  -> trace end
```

## Key Files

| File | Purpose |
|------|---------|
| `policy_ingestion.py` | Project Aegis ingestion engine: markdown chunking, table preservation, metadata extraction, verification, upsert |
| `app/core/vector_store.py` | Shared Chroma store, embedding provider selection, metadata-filtered retrieval |
| `app/nodes/retrieval.py` | Query expansion, HyDE-style retrieval, metadata filters, post-filtering, reranking |
| `app/core/models.py` | Ollama-optional local generation with extractive fallback |
| `app/graph/workflow.py` | Full LangGraph workflow |
| `app/nodes/planner.py` | Rule-based grade-aware router |
| `app/nodes/verifier.py` | Blocking quality gate |
| `streamlit_app.py` | Primary Streamlit chat UI and execution trace viewer |
| `api.py` | FastAPI REST API |
| `tests/test_all.py` | Core unit tests |
| `tests/test_vector_wiring.py` | Ingestion/retrieval wiring regression tests |

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

Drop `.txt`, `.md`, or `.pdf` files anywhere under `data/`. Streamlit auto-indexes them into the shared `db` Chroma store on startup, and `python policy_ingestion.py` can rebuild the index manually.
