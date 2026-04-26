import streamlit as st

# ✅ STRICT ALLOWED MODELS

ALLOWED_LLM_MODELS = {
"gpt-5-nano",
"gpt-5-mini",
"gpt-4o-mini"
}

ALLOWED_EMBED_MODELS = {
"text-embedding-3-large",
"text-embedding-3-small"
}

def get_llm_model():
model = st.secrets.get("MODEL_NAME")

```
if not model:
    raise ValueError("MODEL_NAME not set in Streamlit secrets")

if model not in ALLOWED_LLM_MODELS:
    raise ValueError(f"❌ Unauthorized LLM model: {model}")

return model
```

def get_embedding_model():
model = st.secrets.get("EMBED_MODEL")

```
if not model:
    raise ValueError("EMBED_MODEL not set in Streamlit secrets")

if model not in ALLOWED_EMBED_MODELS:
    raise ValueError(f"❌ Unauthorized embedding model: {model}")

return model
```
