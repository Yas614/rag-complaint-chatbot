# app.py
import streamlit as st
from src.rag_engine import CrediTrustRAG

# Page Configuration for Premium Look and Feel
st.set_page_config(
    page_title="CrediTrust Feedback Agent",
    page_icon="🛡️",
    layout="wide"
)

# Cache RAG Engine Instance so it doesn't reload on every UI click
@st.cache_resource
def load_engine():
    return CrediTrustRAG()

try:
    rag = load_engine()
except Exception as e:
    st.error(f"Failed to initialize ChromaDB Engine: {e}. Please ensure src/chunk_embed.py ran successfully.")
    st.stop()

# Header Layout
st.title("🛡️ CrediTrust Financial Analytics Engine")
st.subheader("Intelligent RAG Portal for Customer Complaint Insights")
st.markdown("---")

# Sidebar Filters - Empowers Support & Compliance to sort effortlessly
st.sidebar.header("🎛️ Control Panel")
product_scope = st.sidebar.selectbox(
    "Target Business Operational Segment Filter",
    ["All", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]
)

st.sidebar.markdown("""
### How to Query:
1. Select a specific product department to isolate queries, or choose **All**.
2. Type plain English questions like: 
   * *'Why are individuals unhappy with credit card limits?'*
   * *'What processing hurdles occur during money transfers?'*
""")

# Main Chat Input Interface
query_input = st.text_input("💬 Ask an analytical question about real customer complaint trends:", placeholder="Type your query here...")

if query_input:
    with st.spinner("🔍 Executing semantic index retrieval and generating grounded response..."):
        answer, sources = rag.query(query_input, product_filter=product_scope)
        
        # Display Generated Answer Component
        st.markdown("### 🤖 Synthesized Intelligence Response")
        st.info(answer)
        
        # Display Sources Component (Mandatory Trust & Verification Requirement)
        st.markdown("### 📋 Evidence Footprints (Source Document Excerpts)")
        
        for i, src in enumerate(sources):
            with st.expander(f"📄 Source Chunk {i+1} [Complaint ID: {src['id']}] - Category: {src['product']}"):
                st.write(f"*{src['text']}*")

# Reset Conversation Button Functionality
if st.sidebar.button("🔄 Clear Active Session"):
    st.rerun()