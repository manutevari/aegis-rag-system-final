"""Main entry point — CLI / API / UI modes."""
import argparse, asyncio, logging, os
from dotenv import load_dotenv; load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")

def run_cli():
    from app.graph.workflow import build_graph
    from app.utils.pickle_cache import PickleCache
    from app.utils.encryption import encrypt, decrypt
    cache = PickleCache(); graph = build_graph()
    print("\n🏢  Decision-Grade Policy RAG  (type 'exit' to quit)\n")
    while True:
        q = input("Query > ").strip()
        if q.lower() in ("exit","quit"): break
        if not q: continue
        b = cache.get(q)
        if b: print(f"\n[CACHE]\n{decrypt(b)}\n"); continue
        result = asyncio.run(graph.ainvoke({"query":q,"history":[],"trace_log":[],"retry_count":0}))
        ans = result.get("answer","No answer.")
        cache.set(q, encrypt(ans))
        print(f"\n{ans}\n")

def run_api():
    import uvicorn; uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)

def run_ui():
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/ui/streamlit_app.py"])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["cli","api","ui"], default="ui")
    args = p.parse_args()
    {"cli": run_cli, "api": run_api, "ui": run_ui}[args.mode]()
