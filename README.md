# Decision-Grade Corporate Policy RAG Chatbot

A production-grade, context-aware RAG chatbot for navigating complex, interconnected, and highly numerical corporate policy, built on LangGraph, LangChain, FAISS/Chroma, FastAPI, and Streamlit.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY

# Streamlit UI -> http://localhost:8501
streamlit run streamlit_app.py

# REST API -> http://localhost:8000/docs
uvicorn api:app --reload --host 0.0.0.0 --port 8000
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
| `main.py` | Alternate Streamlit upload/rerank prototype |
| `api.py` | FastAPI REST API |
| `app/graph/workflow.py` | Full LangGraph workflow |
| `app/nodes/planner.py` | LLM + keyword grade-aware router |
| `app/nodes/verifier.py` | Blocking quality gate |
| `app/tools/compute.py` | Pure Python arithmetic helpers |
| `app/tools/sql.py` | SQLite/PostgreSQL policy database |
| `app/tools/retriever.py` | FAISS/Chroma vector search |
| `tests/test_all.py` | Unit tests |

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

Drop `.txt`, `.md`, or `.pdf` files into `data/policies/`. They are auto-chunked and indexed on startup. Built-in sample policies (Travel T-04, Leave HR-07, Compensation C-03, IT IT-09, Reimbursement F-12) work out of the box.
