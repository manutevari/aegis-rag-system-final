from typing import List, Dict
import os

# Simple in-memory store (can swap to Redis later)
class ConversationBuffer:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get(self, k: int = 5):
        return self.history[-k:]

    def format(self) -> str:
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.get()])


# Vector Memory (Long-term)
class VectorMemory:
    def __init__(self, embedding_model, vectorstore):
        self.embedding_model = embedding_model
        self.vs = vectorstore

    def add(self, text: str):
        self.vs.add_texts([text])

    def search(self, query: str, k: int = 3):
        return self.vs.similarity_search(query, k=k)