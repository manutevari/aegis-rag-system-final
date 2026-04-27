MAX_TOTAL = 500
MAX_CTX = 220
MAX_PROMPT = 160
MAX_OUT = 120

def _words(s): return s.split()

def trim_words(s: str, max_tokens: int) -> str:
    return " ".join(_words(s)[:max_tokens])

def build_context(chunks, per_chunk=110, k=2):
    # take top-k and trim each
    sel = chunks[:k]
    trimmed = [trim_words(c, per_chunk) for c in sel]
    ctx = "\n\n".join(trimmed)
    return trim_words(ctx, MAX_CTX)

def build_prompt(system: str, query: str, context: str) -> str:
    prompt = f"{system}\n\nCONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nANSWER:"
    return trim_words(prompt, MAX_PROMPT)
