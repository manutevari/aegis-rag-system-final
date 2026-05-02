# Decision-Grade Corporate Policy RAG Chatbot

A production-grade, context-aware RAG chatbot for navigating complex, interconnected, and highly numerical corporate policy documents. The implementation follows the Project Aegis guidelines for semantic ingestion, metadata-aware retrieval, query transformation, reranking, and grounded answer verification.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env .env.local

# Local fallback mode, no hosted keys required
RAG_EMBEDDINGS_PROVIDER=hash VECTOR_BACKEND=chroma LLM_PROVIDER=extractive python policy_ingestion.py
RAG_EMBEDDINGS_PROVIDER=hash VECTOR_BACKEND=chroma LLM_PROVIDER=extractive uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Hosted Provider Stack

AEGIS is configured for this production stack:

- OpenAI `gpt-4o-mini` for grounded answer generation.
- OpenAI `text-embedding-3-large` for dense embeddings.
- Pinecone for metadata-filtered vector storage and retrieval.
- Cohere `rerank-v3.5` for final context reranking.
- Pydantic settings in `app/core/settings.py` for typed provider configuration.

Set these environment variables in `.env.local`, your deployment secrets, or your hosting platform:

```bash
OPENAI_API_KEY=...
COHERE_API_KEY=...
PINECONE_API_KEY=...

LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
RAG_EMBEDDINGS_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072

VECTOR_BACKEND=pinecone
PINECONE_INDEX_NAME=aegis-policies
PINECONE_NAMESPACE=default
# Recommended in production to avoid one extra control-plane lookup:
PINECONE_INDEX_HOST=your-index-host.pinecone.io

RERANK_PROVIDER=cohere
COHERE_RERANK_MODEL=rerank-v3.5
```

Build and run with the hosted stack:

```bash
python policy_ingestion.py
uvicorn api:app --reload --host 0.0.0.0 --port 8000
streamlit run streamlit_app.py --server.fileWatcherType none
```

If any hosted credential is missing, the app falls back safely: OpenAI generation falls back to extractive local answers, OpenAI embeddings fall back to hash embeddings, and Pinecone falls back to Chroma.

## Pinecone Index

For `text-embedding-3-large`, create a dense Pinecone index with:

- dimension: `3072`
- metric: `cosine`
- cloud/region: defaults are `aws` / `us-east-1`

You can let the app create the index in development by setting:

```bash
PINECONE_CREATE_INDEX=true
```

For production, create the index in Pinecone first and set `PINECONE_INDEX_HOST`.

## AEGIS Ingestion Engine

`policy_ingestion.py` implements the guideline ingestion pipeline:

1. Markdown-aware semantic chunking parses `#`, `##`, and `###` headers so each chunk carries its section path.
2. Table preservation keeps Markdown tables intact; oversized tables are chunked by rows with the header repeated.
3. Sequential overlap adds a 10-15% context bridge between neighboring chunks.
4. Metadata extraction attaches `document_id`, `policy_category`, `policy_owner`, `effective_date`, `last_revised`, `h1_header`, `h2_header`, `h3_header`, `source_path`, and `section_path` to every chunk.
5. Ingestion verification blocks indexing if required metadata is missing or a chunk is empty.
6. Upsertion batches the verified chunks into Pinecone with metadata payloads when configured, otherwise Chroma.

## Advanced Retrieval Pipeline

`app/nodes/retrieval.py` implements the guideline retrieval pipeline:

1. Query expansion generates policy-oriented rewrites such as rideshare, cab fare, and ground transportation variants for a taxi question.
2. HyDE-style search text creates a hypothetical policy answer and retrieves against that richer text.
3. Metadata pre-filtering applies category filters such as `policy_category == travel` before vector search.
4. Broad retrieval pools up to 25 chunks across the raw query, expanded queries, and HyDE query.
5. Latest-version post-filtering keeps the newest effective policy version when multiple versions of a policy family are retrieved.
6. Cohere reranking scores pooled chunks and passes only the top 5 into generation, with lexical fallback if no Cohere key is configured.

## Architecture

```text
User query
  -> planner / router
  -> advanced retrieval
       -> query expansion
       -> HyDE-style search text
       -> metadata pre-filter
       -> OpenAI embeddings + Pinecone vector retrieval
       -> latest-version post-filter
       -> Cohere rerank top 5
  -> context assembler / token manager
  -> OpenAI gpt-4o-mini grounded generator
  -> confidence and verifier
  -> retry or HITL fallback
  -> trace end
```

## Key Files

| File | Purpose |
|------|---------|
| `app/core/settings.py` | Pydantic runtime settings for OpenAI, Cohere, Pinecone, Chroma, and local fallbacks |
| `policy_ingestion.py` | Project Aegis ingestion engine: markdown chunking, table preservation, metadata extraction, verification, upsert |
| `app/core/vector_store.py` | OpenAI embeddings, Pinecone vector store, Chroma/hash fallbacks, metadata-filtered retrieval |
| `app/nodes/retrieval.py` | Query expansion, HyDE-style retrieval, metadata filters, post-filtering, Cohere reranking |
| `app/core/models.py` | OpenAI `gpt-4o-mini` generation with extractive/Ollama fallback paths |
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

Drop `.txt`, `.md`, or `.pdf` files anywhere under `data/`. Streamlit auto-indexes them into the configured vector store on startup, and `python policy_ingestion.py` can rebuild the index manually.
