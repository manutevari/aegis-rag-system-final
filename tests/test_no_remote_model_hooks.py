from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "db", ".streamlit", ".venv", "venv"}
REMOTE_TOKEN = "op" + "en" + "ai"
FORBIDDEN = [
    "Open" + "AI",
    "Chat" + "Open" + "AI",
    "OPEN" + "AI_API_KEY",
    "langchain_" + REMOTE_TOKEN,
    "langchain-" + REMOTE_TOKEN,
    REMOTE_TOKEN,
]


def _iter_text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pyc", ".png", ".jpg", ".jpeg", ".pdf", ".sqlite3"}:
            continue
        yield path


def test_no_hosted_model_hooks_remain():
    hits = []
    for path in _iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for needle in FORBIDDEN:
            if needle in text:
                hits.append(f"{path.relative_to(ROOT)}: {needle}")

    assert hits == []
