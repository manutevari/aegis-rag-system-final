import streamlit as st
import sys
import os

# 🔧 Fix Python path for Streamlit Cloud
sys.path.append(os.path.abspath("."))

# 🔧 Page config
st.set_page_config(page_title="AEGIS RAG", layout="wide")

# 🔹 UI Header
st.title("AEGIS RAG System")
st.info("Deployment fixed. Backend loading...")

# 🔧 Safe import handling
run_query = None
import_error = None

try:
    from src.workflow import run_query
except Exception as e1:
    try:
        from workflow import run_query
    except Exception as e2:
        import_error = f"{e1} | {e2}"

# 🔴 If import fails
if run_query is None:
    st.error(f"Backend load error: {import_error}")
    st.stop()

# 🔹 User Input
query = st.text_input("Enter your query")

# 🔹 Execution block
if query:
    with st.spinner("Processing..."):
        try:
            result = run_query(query)
            st.success("Response:")
            st.write(result)
        except Exception as e:
            st.error(f"Execution error: {e}")
