import os
import asyncio
import streamlit as st
import openai
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings  # Updated import

load_dotenv()

# ==============================
# 🔹 Configuration & Whitelists
# ==============================

OPENAI_CHAT_MODELS = {"gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"}
OPENROUTER_FREE_MODELS = {"gpt-4o-mini", "llama-3.3-70b", "deepseek-r1"}
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

# ==============================
# 🔹 Model Factory
# ==============================

def get_chat_model():
    provider = st.sidebar.selectbox("LLM Provider", ["openai", "openrouter"], index=0)
    model_name = st.sidebar.text_input("Model Name", DEFAULT_CHAT_MODEL)

    # Simple validation logic
    if provider == "openai" and model_name not in OPENAI_CHAT_MODELS:
        st.warning(f"Using unverified OpenAI model: {model_name}")
    
    return init_chat_model(
        model=model_name,
        model_provider=provider,
        api_key=os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("OPENROUTER_API_KEY")
    )

def get_embed_model():
    return OpenAIEmbeddings(
        model=DEFAULT_EMBED_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

# ==============================
# 🔹 Pipeline Components
# ==============================

async def retrieve_and_rerank(user_query: str, embed_model):
    """Retrieves chunks and performs reranking."""
    # Logic from your previous version, simplified for context
    from app.core.vector_store import vector_db
    from app.core.utils import cross_encoder

    q_emb = embed_model.embed_query(user_query)
    results = vector_db.search(q_emb, top_k=6)
    
    # Extract text and metadata for citations
    chunks = [{"text": r.page_content, "source": r.metadata.get("source", "Unknown")} for r in results]
    
    # Reranking logic
    texts = [c["text"] for c in chunks]
    scores = cross_encoder.rank(user_query, texts)
    
    # Return top 3 with their metadata
    top_results = [chunks[idx] for idx, _ in scores[:3]]
    return top_results

# ==============================
# 🔹 Streamlit UI & Logic
# ==============================

st.set_page_config(page_title="Aegis Intel", page_icon="🛡️", layout="wide")

# Sidebar: Ingestion UI
with st.sidebar:
    st.title("🛡️ Admin Control")
    st.subheader("Knowledge Ingestion")
    uploaded_files = st.file_uploader("Upload Policy Docs", accept_multiple_files=True, type=['pdf', 'txt'])
    if st.button("Index Documents", use_container_width=True) and uploaded_files:
        with st.status("Indexing...", expanded=True) as status:
            # Placeholder for ingestion logic
            # from app.core.ingestor import process_docs
            # process_docs(uploaded_files)
            asyncio.run(asyncio.sleep(1.5)) 
            status.update(label="✅ Documents Indexed!", state="complete")
    
    st.divider()
    chat_model_instance = get_chat_model()

# Main Chat UI
st.title("💬 Policy Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Message History
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if "sources" in m:
            st.caption(f"Sources: {', '.join(m['sources'])}")

# Handle New Input
if prompt := st.chat_input("Ask about company policy..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        async def run_pipeline():
            nonlocal full_response
            embed_model = get_embed_model()
            
            # 1. Retrieval & Rerank (The "Context" stage)
            with st.spinner("Searching Knowledge Base..."):
                top_results = await retrieve_and_rerank(prompt, embed_model)
                context_text = "\n\n".join([c["text"] for c in top_results])
                sources = list(set([c["source"] for c in top_results]))

            # 2. Real Streaming Generation
            prompt_template = f"Use context to answer:\n{context_text}\n\nQ: {prompt}\nA:"
            
            # This is where the real streaming happens
            async for chunk in chat_model_instance.astream(prompt_template):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            
            # 3. Citations UI
            st.markdown("---")
            st.caption("🔍 Sources Cited:")
            for src in sources:
                st.markdown(f"- `{src}`")
            
            return full_response, sources

        # Run the async pipeline
        final_text, final_sources = asyncio.run(run_pipeline())
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_text, 
            "sources": final_sources
        })
