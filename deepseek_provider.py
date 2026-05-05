from __future__ import annotations

from typing import List, Optional


def apply_deepseek(app) -> None:
    deepseek_config = {
        "kind": "openai",
        "base_url": "https://api.deepseek.com",
        "models": ["deepseek-v4-flash", "deepseek-v4-pro", "deepseek-chat", "deepseek-reasoner", "custom"],
        "env": ["DEEPSEEK_API_KEY", "DEEPSEEK_KEY"],
    }
    updated = {}
    inserted = False
    for name, config in app.PROVIDERS.items():
        if name == "DeepSeek":
            continue
        updated[name] = config
        if name == "Grok":
            updated["DeepSeek"] = deepseek_config
            inserted = True
    if not inserted:
        updated["DeepSeek"] = deepseek_config
    app.PROVIDERS.clear()
    app.PROVIDERS.update(updated)

    def provider_order(question: str, calc: Optional[dict], mood: dict) -> List[str]:
        del question
        order = ["DeepSeek", "Hugging Face", "Gemini", "OpenAI", "Mistral", "Grok", "OpenRouter"]
        if calc:
            order = ["DeepSeek", "Gemini", "OpenAI", "Hugging Face", "Mistral", "Grok", "OpenRouter"]
        if mood["label"] == "negative":
            order = ["OpenAI", "DeepSeek", "Gemini", "Grok", "Hugging Face", "Mistral", "OpenRouter"]
        return order

    app.provider_order = provider_order
