"""
Unified Memory Manager

- Short-term memory (recent conversation)
- Long-term memory (vector store retrieval)
- Context builder for LLM
"""

from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────
# 🔹 Short-Term Memory (Conversation Buffer)
# ─────────────────────────────────────────────────────────────

class ConversationBuffer:
    def __init__(self, max_turns: int = 5):
        self.history: List[Dict[str, str]] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

        # keep only last N turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get(self) -> List[Dict[str, str]]:
        return self.history

    def format(self) -> str:
        return "\n".join(
            f"{m['role']}: {m['content']}" for m in self.history
        )


# ─────────────────────────────────────────────────────────────
# 🔹 Long-Term Memory (Vector Store)
# ─────────────────────────────────────────────────────────────

class VectorMemory:
    def __init__(self, vectorstore):
        self.vs = vectorstore

    def add(self, text: str):
        if text:
            self.vs.add_texts([text])

    def search(self, query: str, k: int = 3) -> List[str]:
        results = self.vs.similarity_search(query, k=k)
        return [getattr(r, "page_content", str(r)) for r in results]


# ─────────────────────────────────────────────────────────────
# 🔹 Unified Memory Manager
# ─────────────────────────────────────────────────────────────

class MemoryManager:
    def __init__(self, vectorstore=None, max_turns: int = 5):
        self.buffer = ConversationBuffer(max_turns=max_turns)
        self.vector = VectorMemory(vectorstore) if vectorstore else None

    # --------------------------
    # Add interaction
    # --------------------------
    def add(self, user: str, assistant: str):
        # short-term memory
        self.buffer.add("User", user)
        self.buffer.add("Assistant", assistant)

        # long-term memory (store only meaningful responses)
        if self.vector and len(assistant) > 20:
            self.vector.add(f"User: {user}\nAssistant: {assistant}")

    # --------------------------
    # Build context for LLM
    # --------------------------
    def get_context(self, query: str = "") -> str:
        short_term = self.buffer.format()

        long_term = ""
        if self.vector and query:
            retrieved = self.vector.search(query, k=3)
            long_term = "\n".join(retrieved)

        return f"""
# Recent Conversation:
{short_term}

# Relevant Past Memory:
{long_term}
""".strip()

    # --------------------------
    # Debug helpers
    # --------------------------
    def get_recent(self) -> List[Dict[str, str]]:
        return self.buffer.get()

    def clear(self):
        self.buffer.history = []
