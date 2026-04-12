# =============================================================================
# AEGIS — CENTRALIZED STRUCTURED LOGGER
# logger.py
#
# Provides structured JSON logging for every pipeline stage.
# Every node, fallback, error, and decision is written to:
#   • Console (human-readable)
#   • aegis_pipeline.log (machine-parseable JSON, one event per line)
#
# Log levels:
#   DEBUG   — node inputs/outputs, intermediate scores
#   INFO    — node entry/exit, pipeline milestones
#   WARNING — fallbacks triggered, degraded paths taken
#   ERROR   — exceptions caught, partial failures
#   CRITICAL — full pipeline failure (graph crash)
#
# Usage (every module):
#   from logger import get_logger
#   log = get_logger(__name__)
#   log.info("node_retrieve entered", extra={"query": query, "top_k": 25})
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# JSON formatter — every log record emitted as one JSON line
# ---------------------------------------------------------------------------

class _JSONFormatter(logging.Formatter):
    """
    Converts a LogRecord into a single JSON line for structured log ingestion.
    Fields: timestamp, level, logger, message, + any extra kwargs.
    """

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        # Merge any extra fields passed via extra={...}
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno",
                "pathname", "filename", "module", "exc_info",
                "exc_text", "stack_info", "lineno", "funcName",
                "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "message",
            ):
                base[key] = val
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Console formatter — readable for human operators
# ---------------------------------------------------------------------------

class _ConsoleFormatter(logging.Formatter):
    COLOURS = {
        "DEBUG":    "\033[36m",   # cyan
        "INFO":     "\033[32m",   # green
        "WARNING":  "\033[33m",   # yellow
        "ERROR":    "\033[31m",   # red
        "CRITICAL": "\033[35m",   # magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.COLOURS.get(record.levelname, "")
        ts     = datetime.now(timezone.utc).strftime("%H:%M:%S")
        prefix = f"{colour}[{record.levelname[:4]}]{self.RESET}"
        return f"{ts} {prefix} {record.name.split('.')[-1]}: {record.getMessage()}"


# ---------------------------------------------------------------------------
# Logger registry — one logger per module, shared handlers
# ---------------------------------------------------------------------------

_LOG_FILE    = os.getenv("AEGIS_LOG_FILE", "aegis_pipeline.log")
_LOG_LEVEL   = os.getenv("AEGIS_LOG_LEVEL", "INFO").upper()
_initialized = False
_registry: dict[str, logging.Logger] = {}


def _init_root() -> None:
    global _initialized
    if _initialized:
        return

    root = logging.getLogger("aegis")
    root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))
    root.propagate = False

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_ConsoleFormatter())
    root.addHandler(ch)

    # File handler (JSON)
    try:
        fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
        fh.setFormatter(_JSONFormatter())
        root.addHandler(fh)
    except Exception:
        pass  # If log file can't be created, console only

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger scoped under the 'aegis.' hierarchy.
    Call once per module: log = get_logger(__name__)
    """
    _init_root()
    full_name = f"aegis.{name}" if not name.startswith("aegis.") else name
    if full_name not in _registry:
        logger = logging.getLogger(full_name)
        _registry[full_name] = logger
    return _registry[full_name]


# ---------------------------------------------------------------------------
# Pipeline-specific helpers
# ---------------------------------------------------------------------------

class PipelineLogger:
    """
    Wrapper that attaches structured fields to every log call,
    tying log lines to a specific query run.
    """

    def __init__(self, query: str, run_id: str | None = None) -> None:
        self._log    = get_logger("pipeline")
        self._query  = query[:80]
        self._run_id = run_id or f"run_{int(time.time() * 1000) % 1_000_000}"
        self._start  = time.perf_counter()

    def _extra(self, **kw) -> dict:
        return {"run_id": self._run_id, "query": self._query, **kw}

    def node_enter(self, node: str, **kw) -> None:
        self._log.info(
            f"→ {node} entered",
            extra=self._extra(node=node, event="node_enter", **kw),
        )

    def node_exit(self, node: str, **kw) -> None:
        elapsed = round(time.perf_counter() - self._start, 3)
        self._log.info(
            f"← {node} done ({elapsed}s)",
            extra=self._extra(node=node, event="node_exit", elapsed_s=elapsed, **kw),
        )

    def fallback(self, node: str, reason: str, **kw) -> None:
        self._log.warning(
            f"⚠ FALLBACK in {node}: {reason}",
            extra=self._extra(node=node, event="fallback", reason=reason, **kw),
        )

    def error(self, node: str, exc: Exception, **kw) -> None:
        self._log.error(
            f"✗ ERROR in {node}: {exc}",
            extra=self._extra(node=node, event="error",
                              exception=str(exc),
                              traceback=traceback.format_exc()[-800:],
                              **kw),
        )

    def metric(self, name: str, value: Any, **kw) -> None:
        self._log.debug(
            f"📊 {name}={value}",
            extra=self._extra(event="metric", metric_name=name,
                              metric_value=value, **kw),
        )

    def pipeline_done(self, answer_chars: int, sources: int) -> None:
        elapsed = round(time.perf_counter() - self._start, 3)
        self._log.info(
            f"✅ Pipeline complete in {elapsed}s "
            f"(answer={answer_chars}chars, sources={sources})",
            extra=self._extra(event="pipeline_done",
                              total_elapsed_s=elapsed,
                              answer_chars=answer_chars,
                              sources_count=sources),
        )
