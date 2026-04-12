# =============================================================================
# AEGIS — UTILITY FUNCTIONS
# =============================================================================

from __future__ import annotations

import re
import textwrap
import unicodedata


def clean_text(text: str) -> str:
    """
    Normalize whitespace and remove control characters from raw document text.
    Preserves Markdown structure (headers, tables, code fences).
    """
    if not text:
        return ""
    # Normalize unicode (NFC)
    text = unicodedata.normalize("NFC", text)
    # Collapse runs of spaces/tabs (but not newlines — needed for Markdown)
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ consecutive blank lines to two
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars for display, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def format_date(iso_date: str) -> str:
    """Format ISO date string (YYYY-MM-DD) to human-readable form."""
    try:
        from datetime import date
        d = date.fromisoformat(iso_date)
        return d.strftime("%B %d, %Y")
    except Exception:
        return iso_date


def load_txt_file(path: str) -> str:
    """Read a UTF-8 text file, with fallback to latin-1."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()


def chunk_stats(chunks: list) -> dict:
    """Return basic statistics about a list of ChunkMetadata objects."""
    if not chunks:
        return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
    lengths = [len(c.chunk_text) for c in chunks]
    return {
        "count":   len(chunks),
        "avg_len": int(sum(lengths) / len(lengths)),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }
