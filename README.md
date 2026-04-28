# Decision-Grade Corporate Policy RAG Chatbot

A production-grade, context-aware RAG chatbot for navigating complex, interconnected, and highly numerical corporate policy, built on LangGraph, LangChain, Chroma, FastAPI, and Streamlit.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY when you want OpenAI embeddings/LLM

# Optional: build the shared Chroma policy index manually
python policy_ingestion.py

# Streamlit UI -> http://localhost:8501
streamlit run streamlit_app.py

# REST API -> http://localhost:8000/docs
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## Offline Embeddings

The vector layer is offline-capable. With `OPENAI_API_KEY`, it uses OpenAI embeddings by default. Without a key, it tries cached local Hugging Face embeddings and falls back to deterministic local hash embeddings, so ingestion and retrieval can still run without internet.

```bash
# Guaranteed no-download local mode
RAG_EMBEDDINGS_PROVIDER=hash python policy_ingestion.py

# Prefer cached local Hugging Face embeddings, then fallback to hash
RAG_EMBEDDINGS_PROVIDER=local python policy_ingestion.py

# Prefer local mode even if OPENAI_API_KEY is present
RAG_OFFLINE_FIRST=true streamlit run streamlit_app.py
```

## Architecture

```text
Query -> Planner (grade-aware routing)
  |-- sql -> compute -> context_assembler
  |-- retrieval ------> context_assembler
  `-- direct ---------> context_assembler
                         |
                   token_check (summarise if >3k tokens)
                         |
                     generate (strict grounded LLM)
                         |
                      verify (4 blocking checks)
                    valid |       retry
                         v         ^ 
                       hitl -------'
                         |
                   encrypt -> decrypt -> trace -> END
```

## Key Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Primary Streamlit chat UI and execution trace viewer |
| `policy_ingestion.py` | Canonical ingestion entrypoint for policy files |
| `app/core/vector_store.py` | Single Chroma store authority for ingestion and retrieval |
| `api.py` | FastAPI REST API |
| `app/graph/workflow.py` | Full LangGraph workflow |
| `app/nodes/planner.py` | LLM + keyword grade-aware router |
| `app/nodes/retrieval.py` | Graph retrieval node using the shared vector store |
| `app/nodes/verifier.py` | Blocking quality gate |
| `app/tools/compute.py` | Pure Python arithmetic helpers |
| `app/tools/sql.py` | SQLite/PostgreSQL policy database |
| `tests/test_all.py` | Core unit tests |
| `tests/test_vector_wiring.py` | Ingestion/retrieval wiring regression tests |

## Run Tests

```bash
pytest tests/ -v
```

## Docker

```bash
cp .env.example .env
docker-compose up -d
# UI: http://localhost:8501 | API: http://localhost:8000/docs
```

## Add Your Own Policies

Drop `.txt`, `.md`, or `.pdf` files anywhere under `data/`. Streamlit auto-indexes them into the shared `db` Chroma store on startup, and `python policy_ingestion.py` can rebuild the index manually.
