"""
Evaluation — Pandas + Plotly metrics over execution traces.
Run: python -m app.evaluation.metrics
"""
import json, logging, os, pathlib
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

logger    = logging.getLogger(__name__)
TRACE_DIR = pathlib.Path(os.getenv("TRACE_DIR", "/tmp/dg_rag_traces"))

def load_traces(limit=500) -> List[dict]:
    if not TRACE_DIR.exists(): return []
    files = sorted(TRACE_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    out = []
    for f in files[:limit]:
        try: out.append(json.loads(f.read_text()))
        except: pass
    return out

def evaluate(logs: List[dict]) -> pd.DataFrame:
    rows = []
    for log in logs:
        steps = log.get("steps", [])
        latency = ((steps[-1]["ts"] - steps[0]["ts"]) * 1000) if len(steps) >= 2 else 0
        rows.append({
            "ts": log.get("ts",0), "query": log.get("query",""),
            "route": log.get("route","?"), "grade": log.get("grade",""),
            "verified": bool(log.get("verified",False)),
            "hitl": log.get("hitl","auto"), "retries": int(log.get("retries",0)),
            "tokens": int(log.get("tokens",0)), "latency_ms": round(latency,1),
        })
    return pd.DataFrame(rows)

def summary(df: pd.DataFrame) -> dict:
    if df.empty: return {}
    return {
        "total": len(df),
        "verification_rate_pct": round(df["verified"].mean()*100, 1),
        "avg_latency_ms": round(df["latency_ms"].mean(), 1),
        "p95_latency_ms": round(df["latency_ms"].quantile(0.95), 1),
        "avg_retries": round(df["retries"].mean(), 2),
        "routes": df["route"].value_counts().to_dict(),
    }

def plot_metrics(df: pd.DataFrame, out_dir: str = "/tmp/dg_rag_eval"):
    try: import plotly.express as px
    except ImportError: logger.warning("plotly not installed"); return {}
    import pathlib; pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    figs = {
        "latency":  px.histogram(df, x="latency_ms", title="Latency (ms)"),
        "routes":   px.pie(values=df["route"].value_counts().values, names=df["route"].value_counts().index, title="Routes"),
        "verified": px.bar(df.groupby("route")["verified"].mean().reset_index(), x="route", y="verified", title="Verification Rate by Route"),
    }
    for name, fig in figs.items():
        fig.write_html(f"{out_dir}/{name}.html")
    return figs

def numerical_accuracy(questions, answers, ground_truth, tol=1.0):
    import re; _N = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
    correct, results = 0, []
    for q, a, gt in zip(questions, answers, ground_truth):
        nums = [float(m.replace(",","")) for m in _N.findall(a)]
        hit  = any(abs(n-gt)<=tol for n in nums)
        if hit: correct += 1
        results.append({"query":q,"expected":gt,"found":nums,"correct":hit})
    return {"accuracy_pct": round(correct/len(questions)*100,1) if questions else 0, "details": results}

if __name__ == "__main__":
    traces = load_traces()
    if not traces: print("No traces yet — run some queries first."); exit()
    df = evaluate(traces)
    print(df.describe())
    print(summary(df))
    plot_metrics(df)
    print("Plots saved to /tmp/dg_rag_eval/")
