import streamlit as st

st.set_page_config(page_title="AEGIS RAG")

st.title("AEGIS RAG System")

st.info("Deployment fixed. Backend will load below.")

try:
    from src.workflow import run_query
    query = st.text_input("Enter your query")
    if query:
        result = run_query(query)
        st.write(result)
except Exception as e:
    st.error(f"Backend load error: {e}")
