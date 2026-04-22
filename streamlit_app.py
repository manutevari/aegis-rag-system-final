from __future__ import annotations

# =============================================================================
# AEGIS — ENTERPRISE RAG SYSTEM  |  Streamlit UI  (LangGraph Edition)
# =============================================================================

# Standard library
import os
import datetime
import json
import re
import tempfile
import time

# Third-party
import streamlit as st

# 🔑 Load secrets (only once)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]

# other imports (langchain, pinecone, etc.)

from utils import clean_text, load_txt_file, truncate, format_date

st.set_page_config(
    page_title="Aegis — Enterprise RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"]          { background: #161b27; border-right: 1px solid #2a2f3d; }
.aegis-header { display:flex;align-items:center;gap:14px;padding:18px 0 12px; }
.aegis-logo   { font-size:38px;line-height:1; }
.aegis-title  { font-size:26px;font-weight:700;letter-spacing:-0.5px;color:#e8eaf0;margin:0; }
.aegis-subtitle { font-size:13px;color:#6b7280;margin:0; }
.status-badge { display:inline-flex;align-items:center;gap:6px;padding:4px 10px;
                border-radius:20px;font-size:12px;font-weight:500; }
.status-ok   { background:#0d2818;color:#4ade80;border:1px solid #166534; }
.status-warn { background:#2d1a0a;color:#fb923c;border:1px solid #7c2d12; }
.msg-user { background:#1e2535;border:1px solid #2a3045;border-radius:12px;
            padding:14px 18px;margin:8px 0;color:#d1d5db;font-size:15px; }
.msg-assistant { background:#111827;border:1px solid #1f2937;border-radius:12px;
                 padding:16px 20px;margin:8px 0;color:#e5e7eb;font-size:15px;line-height:1.7; }
.msg-role { font-size:11px;font-weight:600;letter-spacing:0.8px;
            text-transform:uppercase;margin-bottom:6px; }
.msg-role-user { color:#6366f1; }
.msg-role-bot  { color:#10b981; }
.source-card { background:#161b27;border:1px solid #2a3045;border-radius:8px;
               padding:12px 16px;margin:8px 0; }
.source-header { display:flex;justify-content:space-between;align-items:center;margin-bottom:8px; }
.source-meta   { font-size:11px;color:#6b7280;display:flex;gap:12px;flex-wrap:wrap; }
.source-meta span { display:flex;align-items:center;gap:4px; }
.source-score  { font-size:12px;font-weight:600;padding:2px 8px;border-radius:12px;
                 background:#0d2818;color:#4ade80;border:1px solid #166534; }
.source-text   { font-size:13px;color:#9ca3af;line-height:1.6;
                 border-top:1px solid #2a3045;padding-top:8px;margin-top:4px; }
.metric-row  { display:flex;gap:12px;margin:12px 0; }
.metric-card { flex:1;background:#161b27;border:1px solid #2a3045;border-radius:8px;
               padding:12px 14px;text-align:center; }
.metric-val  { font-size:22px;font-weight:700;color:#e8eaf0; }
.metric-lbl  { font-size:11px;color:#6b7280;margin-top:2px; }
.cat-pill    { display:inline-block;padding:2px 10px;border-radius:12px;
               font-size:11px;font-weight:600;letter-spacing:0.4px; }
.cat-Travel  { background:#0c2a3e;color:#38bdf8;border:1px solid #0369a1; }
.cat-HR      { background:#2d1a3d;color:#c084fc;border:1px solid #7e22ce; }
.cat-Finance { background:#1a2d1a;color:#4ade80;border:1px solid #166534; }
.cat-Legal   { background:#2d2a0c;color:#fbbf24;border:1px solid #92400e; }
.cat-IT      { background:#0c1a2d;color:#60a5fa;border:1px solid #1e40af; }
.cat-General { background:#1f2937;color:#9ca3af;border:1px solid #374151; }
.ae-divider  { border:none;border-top:1px solid #2a3045;margin:16px 0; }
.pii-badge   { display:inline-flex;align-items:center;gap:5px;padding:2px 8px;
               border-radius:12px;font-size:10px;font-weight:600;background:#1a2d1a;
               color:#4ade80;border:1px solid #166534;margin-left:8px;vertical-align:middle; }
.filter-active { display:inline-flex;align-items:center;gap:4px;padding:2px 8px;
                 border-radius:12px;font-size:10px;background:#2d1a0a;
                 color:#fb923c;border:1px solid #7c2d12; }
/* Audit log styles */
.audit-entry { font-family:monospace;font-size:11px;color:#6b7280;
               background:#0d1117;border:1px solid #1f2937;border-radius:6px;
               padding:8px 10px;margin:4px 0; }
.audit-tool   { color:#60a5fa;font-weight:600; }
.audit-reason { color:#9ca3af; }
.msg-type-system { color:#f59e0b;font-size:10px;font-weight:700;letter-spacing:0.6px; }
.msg-type-ai     { color:#10b981;font-size:10px;font-weight:700;letter-spacing:0.6px; }
.msg-type-tool   { color:#60a5fa;font-size:10px;font-weight:700;letter-spacing:0.6px; }
.msg-type-human  { color:#6366f1;font-size:10px;font-weight:700;letter-spacing:0.6px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PII Redaction
# Incorporated from simplified version — screens answers before display.
# ---------------------------------------------------------------------------

def redact_pii(text: str) -> str:
    """
    Redact common PII patterns from answer text before display.
    Currently handles: 10-digit phone numbers, email addresses.
    Extend the patterns here to meet your compliance requirements.
    """
    text = re.sub(r"\b\d{10}\b", "[REDACTED PHONE]", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", "[REDACTED EMAIL]", text)
    return text


def _pii_was_redacted(original: str, redacted: str) -> bool:
    """Returns True if redact_pii changed anything."""
    return original != redacted


# ---------------------------------------------------------------------------
# File-based Audit Logging  (audit.json, JSONL format)
# Incorporated from simplified version — persists every query/answer pair.
# ---------------------------------------------------------------------------

AUDIT_LOG_FILE = "audit.json"


def log_audit(query: str, result: dict) -> None:
    """
    Append a structured audit record to audit.json (one JSON object per line).
    Captures: timestamp, query, answer excerpt, category, source metadata,
    cache_hit, and hallucination_risk flag.
    Never raises — logging failures must not crash the user-facing app.
    """
    entry = {
        "time":              datetime.datetime.utcnow().isoformat() + "Z",
        "query":             query,
        "answer":            result.get("answer", "")[:500],
        "category":          result.get("category"),
        "retrieved":         result.get("retrieved", 0),
        "sources": [
            {
                "document_id":    s.get("document_id", ""),
                "policy_owner":   s.get("policy_owner", ""),
                "effective_date": s.get("effective_date", ""),
                "rerank_score":   round(s.get("rerank_score", 0.0), 4),
            }
            for s in result.get("sources", [])
        ],
        "cache_hit":          result.get("cache_hit", False),
        "hallucination_risk": result.get("hallucination_risk", False),
        "rrf_applied":        result.get("rrf_applied", False),
    }
    try:
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _load_recent_audit_entries(n: int = 10) -> list[dict]:
    """Load the most recent N entries from audit.json for sidebar display."""
    try:
        with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [json.loads(ln) for ln in lines[-n:]][::-1]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Source post-filtering  (author / effective date)
# Incorporated from simplified version — applied client-side after retrieval.
# ---------------------------------------------------------------------------

def apply_source_filters(
    sources: list[dict],
    author_filter: str,
    date_filter: str,
) -> list[dict]:
    """
    Client-side filter on returned source chunks.
    Both filters are ANDed when set.

    Falls back to original list if no sources survive filtering, so the user
    always sees an answer rather than a blank sources panel.
    """
    if not author_filter and not date_filter:
        return sources

    filtered = [
        s for s in sources
        if (not author_filter or s.get("policy_owner", "") == author_filter)
        and (not date_filter  or s.get("effective_date", "") == date_filter)
    ]
    return filtered if filtered else sources


# ---------------------------------------------------------------------------
# Safe imports
# ---------------------------------------------------------------------------
pipeline_ok  = True
pipeline_err = ""

try:
    from ingestion import ingest_document
    from graph     import run_query
except Exception as exc:
    pipeline_ok  = False
    pipeline_err = str(exc)
    def ingest_document(text):
        return {"error": pipeline_err, "messages": []}
    def run_query(query):
        return {"answer": f"Pipeline unavailable: {pipeline_err}",
                "category": None, "retrieved": 0, "sources": [], "messages": []}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key, default in [
    ("messages",      []),
    ("last_metrics",  {}),
    ("ingested_docs", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Audit message renderer helper  (defined early — used in sidebar AND chat)
# ---------------------------------------------------------------------------
def _render_audit_messages(messages: list) -> None:
    """Render a list of BaseMessage objects as a pipeline audit trail."""
    import json
    from langchain_core.messages import (
        SystemMessage, HumanMessage, AIMessage, ToolMessage
    )
    for msg in messages:
        if isinstance(msg, SystemMessage):
            st.markdown(f"<div class='audit-entry'>"
                        f"<span class='msg-type-system'>SYSTEM</span> "
                        f"{msg.content[:200]}</div>", unsafe_allow_html=True)
        elif isinstance(msg, HumanMessage):
            st.markdown(f"<div class='audit-entry'>"
                        f"<span class='msg-type-human'>HUMAN</span> "
                        f"{msg.content[:200]}</div>", unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.markdown(f"<div class='audit-entry'>"
                        f"<span class='msg-type-ai'>AI</span> "
                        f"{msg.content[:200]}</div>", unsafe_allow_html=True)
        elif isinstance(msg, ToolMessage):
            try:
                data   = json.loads(msg.content)
                tool   = data.get("tool", "?")
                reason = data.get("reason", "")[:180]
                outs   = str(data.get("outputs", ""))[:200]
                st.markdown(
                    f"<div class='audit-entry'>"
                    f"<span class='msg-type-tool'>TOOL</span> "
                    f"<span class='audit-tool'>{tool}</span> — "
                    f"<span class='audit-reason'>{reason}</span><br>"
                    f"<span style='color:#374151'>↳ {outs}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            except Exception:
                st.markdown(f"<div class='audit-entry'>"
                            f"<span class='msg-type-tool'>TOOL</span> "
                            f"{msg.content[:200]}</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;padding:10px 0 6px'>
          <div style='font-size:36px'>🛡️</div>
          <div style='font-size:16px;font-weight:700;color:#e8eaf0'>AEGIS</div>
          <div style='font-size:11px;color:#6b7280'>Enterprise RAG · LangGraph Edition</div>
        </div>
    """, unsafe_allow_html=True)

    badge = ('<span class="status-badge status-ok">● Pipeline ready</span>'
             if pipeline_ok
             else f'<span class="status-badge status-warn">⚠ {pipeline_err[:60]}</span>')
    st.markdown(badge, unsafe_allow_html=True)
    st.markdown("<hr class='ae-divider'>", unsafe_allow_html=True)

    # ── Ingest ──────────────────────────────────────────────────────────
    st.markdown("<div style='font-size:11px;font-weight:700;letter-spacing:1px;"
                "text-transform:uppercase;color:#6b7280;margin-bottom:8px'>"
                "📄 Ingest document</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload policy document", type=["txt", "md"],
        help="Plain-text or Markdown policy file.",
        label_visibility="collapsed",
    )
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        with st.spinner("Ingesting…"):
            raw    = load_txt_file(tmp_path)
            text   = clean_text(raw)
            result = ingest_document(text)

        if "error" not in result:
            cat = result.get("category", "General")
            st.success(f"✓ Ingested **{result['chunks']}** chunks")
            st.markdown(f"""
                <div class='source-card' style='margin-top:4px'>
                  <div class='source-meta'>
                    <span>🆔 {result['document_id']}</span>
                    <span class='cat-pill cat-{cat}'>{cat}</span>
                    <span>📦 {result['upserted']} vectors</span>
                  </div>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.ingested_docs.append(result["document_id"])
            if result.get("messages"):
                with st.expander("🔍 Ingestion audit log"):
                    _render_audit_messages(result["messages"])
        else:
            st.error(f"Ingestion failed: {result['error']}")

    if st.session_state.ingested_docs:
        st.markdown("<div style='font-size:11px;color:#6b7280;font-weight:700;"
                    "letter-spacing:1px;text-transform:uppercase;margin-top:14px'>"
                    "INDEXED DOCUMENTS</div>", unsafe_allow_html=True)
        for doc_id in st.session_state.ingested_docs[-5:]:
            st.markdown(f"<div style='font-size:12px;color:#6b7280;padding:2px 0'>"
                        f"· {doc_id}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='ae-divider'>", unsafe_allow_html=True)

    # ── Source Filters (incorporated from simplified version) ────────────
    st.markdown("<div style='font-size:11px;font-weight:700;letter-spacing:1px;"
                "text-transform:uppercase;color:#6b7280;margin-bottom:8px'>"
                "🔎 Filter Sources</div>", unsafe_allow_html=True)

    author_filter = st.selectbox(
        "Filter by policy owner",
        ["", "HR", "Finance", "Legal", "IT", "GCT-RM", "GCTEM", "Unknown"],
        help="Restrict displayed sources to a specific policy owner / author.",
        label_visibility="visible",
    )

    date_filter = st.text_input(
        "Effective date (YYYY-MM-DD)",
        placeholder="e.g. 2025-01-01",
        help="Show only sources with this exact effective date.",
    )

    # Validate date format
    _date_valid = (
        not date_filter
        or bool(re.match(r"^\d{4}-\d{2}-\d{2}$", date_filter.strip()))
    )
    if date_filter and not _date_valid:
        st.warning("⚠ Date must be YYYY-MM-DD")
        date_filter = ""

    if author_filter or date_filter:
        parts = []
        if author_filter:
            parts.append(f"owner: {author_filter}")
        if date_filter:
            parts.append(f"date: {date_filter}")
        st.markdown(
            f"<span class='filter-active'>🔶 Active — {' · '.join(parts)}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='ae-divider'>", unsafe_allow_html=True)

    # ── Last query metrics ───────────────────────────────────────────────
    if st.session_state.last_metrics:
        m = st.session_state.last_metrics
        st.markdown("<div style='font-size:11px;color:#6b7280;font-weight:700;"
                    "letter-spacing:1px;text-transform:uppercase'>LAST QUERY METRICS</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
            <div class='metric-row'>
              <div class='metric-card'>
                <div class='metric-val'>{m.get('latency','—')}</div>
                <div class='metric-lbl'>Latency (s)</div>
              </div>
              <div class='metric-card'>
                <div class='metric-val'>{m.get('retrieved',0)}</div>
                <div class='metric-lbl'>Candidates</div>
              </div>
              <div class='metric-card'>
                <div class='metric-val'>{m.get('sources',0)}</div>
                <div class='metric-lbl'>Used chunks</div>
              </div>
            </div>
        """, unsafe_allow_html=True)
        if m.get("category"):
            cat = m["category"]
            st.markdown(f"<div style='font-size:12px;color:#6b7280'>Intent: "
                        f"<span class='cat-pill cat-{cat}'>{cat}</span></div>",
                        unsafe_allow_html=True)
        if m.get("router_confidence"):
            conf  = m["router_confidence"]
            color = "#4ade80" if conf == "high" else "#fb923c"
            st.markdown(f"<div style='font-size:11px;color:{color};margin-top:4px'>"
                        f"Router: {'🔑 keyword match' if conf=='high' else '🤖 LLM classified'}"
                        f"</div>", unsafe_allow_html=True)
        if m.get("pii_redacted"):
            st.markdown(
                "<div style='font-size:11px;color:#4ade80;margin-top:4px'>"
                "🛡 PII redacted from answer</div>",
                unsafe_allow_html=True,
            )
        if m.get("filters_applied"):
            st.markdown(
                "<div style='font-size:11px;color:#fb923c;margin-top:4px'>"
                "🔶 Source filter applied</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='ae-divider'>", unsafe_allow_html=True)

    # ── Recent audit log viewer ──────────────────────────────────────────
    with st.expander("📋 Recent audit log"):
        entries = _load_recent_audit_entries(8)
        if not entries:
            st.markdown(
                "<div style='font-size:11px;color:#6b7280'>No entries yet.</div>",
                unsafe_allow_html=True,
            )
        for e in entries:
            ts     = e.get("time", "")[:19].replace("T", " ")
            q      = e.get("query", "")[:55]
            cat    = e.get("category") or "—"
            hits   = e.get("retrieved", 0)
            h_icon = "⚠" if e.get("hallucination_risk") else "✓"
            c_icon = "💾 " if e.get("cache_hit") else ""
            st.markdown(
                f"<div class='audit-entry'>"
                f"<span style='color:#4ade80'>{ts}</span> {c_icon}"
                f"<br><span style='color:#e5e7eb'>{q}</span>"
                f"<br><span class='cat-pill cat-{cat}' style='font-size:9px'>{cat}</span>"
                f" · {hits} retrieved · {h_icon}"
                f"</div>",
                unsafe_allow_html=True,
            )

    if st.button("🗑️  Clear conversation", use_container_width=True):
        st.session_state.messages     = []
        st.session_state.last_metrics = {}
        st.rerun()

# ---------------------------------------------------------------------------
# MAIN PANEL — Header
# ---------------------------------------------------------------------------
st.markdown("""
    <div class='aegis-header'>
        <div class='aegis-logo'>🛡️</div>
        <div>
            <div class='aegis-title'>AEGIS Enterprise RAG</div>
            <div class='aegis-subtitle'>
                LangGraph pipeline · Cross-encoder reranking · PII redaction · Full audit trail
            </div>
        </div>
    </div>
    <hr class='ae-divider'>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    role         = msg["role"]
    content      = msg["content"]
    sources      = msg.get("sources", [])
    category     = msg.get("category")
    audit        = msg.get("audit_messages", [])
    pii_redacted = msg.get("pii_redacted", False)

    if role == "user":
        st.markdown(f"""
            <div class='msg-user'>
              <div class='msg-role msg-role-user'>You</div>
              {content}
            </div>
        """, unsafe_allow_html=True)
    else:
        cat_html = ""
        if category:
            cat_html = f"<span class='cat-pill cat-{category}' style='margin-left:8px'>{category}</span>"
        pii_html = "<span class='pii-badge'>🛡 PII redacted</span>" if pii_redacted else ""
        st.markdown(f"""
            <div class='msg-assistant'>
              <div class='msg-role msg-role-bot'>Aegis {cat_html}{pii_html}</div>
              {content}
            </div>
        """, unsafe_allow_html=True)

        # Sources
        if sources:
            with st.expander(f"📎 {len(sources)} source chunk{'s' if len(sources)>1 else ''} used"):
                for i, src in enumerate(sources, 1):
                    score = src.get("rerank_score", 0)
                    h1    = src.get("h1_header", "")
                    h2    = src.get("h2_header", "")
                    cat   = src.get("policy_category", "General")
                    date  = format_date(src.get("effective_date", ""))
                    owner = src.get("policy_owner", "")
                    text  = truncate(src.get("chunk_text", ""), 300)
                    is_tbl = "🗃 Table" if src.get("is_table") else ""
                    st.markdown(f"""
                        <div class='source-card'>
                          <div class='source-header'>
                            <div style='font-size:13px;font-weight:600;color:#d1d5db'>
                              Source {i}{f' — {h2}' if h2 else f' — {h1}' if h1 else ''}
                              {f'<span style="color:#f59e0b;margin-left:6px">{is_tbl}</span>' if is_tbl else ''}
                            </div>
                            <div class='source-score'>{score:.3f}</div>
                          </div>
                          <div class='source-meta'>
                            <span>🗂 <span class='cat-pill cat-{cat}'>{cat}</span></span>
                            {'<span>📅 ' + date + '</span>' if date else ''}
                            {'<span>👤 ' + owner + '</span>' if owner else ''}
                          </div>
                          <div class='source-text'>{text}</div>
                        </div>
                    """, unsafe_allow_html=True)

        # Audit trail
        if audit:
            with st.expander("🔍 Pipeline audit trail"):
                _render_audit_messages(audit)

# ---------------------------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------------------------
query = st.chat_input("Ask about any corporate policy…  e.g. 'Can I expense a taxi?'")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Running LangGraph pipeline…"):
        t0      = time.time()
        result  = run_query(query)
        elapsed = round(time.time() - t0, 2)

    # ── PII Redaction (incorporated from simplified version) ─────────────
    raw_answer   = result.get("answer", "")
    safe_answer  = redact_pii(raw_answer)
    pii_redacted = _pii_was_redacted(raw_answer, safe_answer)

    # ── Source Filtering (incorporated from simplified version) ──────────
    all_sources      = result.get("sources", [])
    filtered_sources = apply_source_filters(
        all_sources,
        author_filter=author_filter,
        date_filter=date_filter.strip() if date_filter else "",
    )
    filters_applied = bool(author_filter or date_filter) and (
        len(filtered_sources) != len(all_sources)
    )

    # ── Extract router confidence for sidebar display ────────────────────
    router_confidence = "low"
    try:
        from langchain_core.messages import ToolMessage
        for m in (result.get("messages") or []):
            if isinstance(m, ToolMessage):
                data = json.loads(m.content)
                if data.get("tool") == "router":
                    router_confidence = data["outputs"].get("confidence", "low")
                    break
    except Exception:
        pass

    # ── File-based audit log (incorporated from simplified version) ───────
    log_audit(query, {**result, "answer": safe_answer})
    st.toast("Audit logged ✓", icon="📋")

    # ── Persist to session state ─────────────────────────────────────────
    st.session_state.messages.append({
        "role":           "assistant",
        "content":        safe_answer,
        "sources":        filtered_sources,
        "category":       result.get("category"),
        "audit_messages": result.get("messages", []),
        "pii_redacted":   pii_redacted,
    })

    st.session_state.last_metrics = {
        "latency":           elapsed,
        "retrieved":         result.get("retrieved", 0),
        "sources":           len(filtered_sources),
        "category":          result.get("category"),
        "router_confidence": router_confidence,
        "pii_redacted":      pii_redacted,
        "filters_applied":   filters_applied,
    }

    st.rerun()
