# 🏢 Decision-Grade Corporate Policy RAG Chatbot

A production-grade, context-aware RAG chatbot for navigating complex, interconnected, and highly numerical corporate policy — built on **LangGraph**, **LangChain**, **FAISS/Chroma**, and **Streamlit**.

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set OPENAI_API_KEY
python main.py --mode ui     # Streamlit UI → http://localhost:8501
python main.py --mode api    # REST API    → http://localhost:8000/docs
python main.py --mode cli    # Terminal
```

## Architecture

```
Query → Planner (grade-aware routing)
  ├── sql → compute → context_assembler
  ├── retrieval ───→ context_assembler
  └── direct ──────→ context_assembler
                         ↓
                   token_check (summarise if >3k tokens)
                         ↓
                     generate (strict grounded LLM)
                         ↓
                      verify (4 blocking checks)
                    valid ↓       ↑ retry
                       hitl ──────┘
                         ↓ approve/edit
                   encrypt → decrypt → trace → END
```

## Key Files

| File | Purpose |
|------|---------|
| `app/graph/workflow.py` | Full 14-node LangGraph |
| `app/nodes/planner.py` | LLM + keyword grade-aware router |
| `app/nodes/verifier.py` | 4-check blocking quality gate |
| `app/tools/compute.py` | Pure Python arithmetic (zero LLM) |
| `app/tools/sql.py` | SQLite/PostgreSQL policy database |
| `app/tools/retriever.py` | FAISS/Chroma vector search |
| `app/ui/streamlit_app.py` | Chat UI + execution trace viewer |
| `app/api.py` | FastAPI REST API |
| `tests/test_all.py` | 20+ unit tests |

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
