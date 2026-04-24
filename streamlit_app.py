"""Streamlit RAG Application - Academic City Assistant - Fixed Version"""

import streamlit as st
import sys
import json
import time
from pathlib import Path
from typing import Any, Literal

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend.config import MANUAL_LOGS_PATH, ensure_dirs
from backend.feedback_store import append_feedback, load_feedback_weights
from backend.pipeline import load_store_and_bm25, run_pipeline

# Page configuration
st.set_page_config(
    page_title="Academic City RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.success-message {
    background: #10b981;
    color: white;
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.chunk-card {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background: #f9fafb;
}
.feedback-buttons {
    display: flex;
    gap: 5px;
    margin-top: 10px;
}
.feedback-btn {
    padding: 5px 10px;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    font-size: 12px;
}
.btn-up { background: #10b981; color: white; }
.btn-down { background: #ef4444; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback_message' not in st.session_state:
    st.session_state.feedback_message = None
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'experiment_logs' not in st.session_state:
    st.session_state.experiment_logs = []

def load_index():
    """Load the FAISS index and BM25 index"""
    try:
        store, bm25 = load_store_and_bm25()
        return store, bm25, True
    except Exception as e:
        st.error(f"Index not found. Please run: python scripts/build_index.py")
        return None, None, False

def format_retrieved_chunks(retrieved):
    """Format retrieved chunks for display"""
    formatted = []
    for i, chunk in enumerate(retrieved):
        formatted.append({
            'source_id': chunk['source_id'],
            'text_preview': chunk['text_preview'],
            'scores_line': f"d={chunk['dense_score']:.3f} | b={chunk['bm25_score']:.3f} | h={chunk['hybrid_score']:.3f}",
            'rank': i
        })
    return formatted

def handle_feedback(source_id, label):
    """Handle feedback submission"""
    append_feedback(source_id, label)
    st.session_state.feedback_message = f"Feedback recorded: {label} for {source_id}"

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
        <h1>🎓 Academic City RAG Assistant</h1>
        <p>CS4241 - Introduction to Artificial Intelligence</p>
        <p><em>Chat with Ghana Election Data & Budget Information</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Load index
    store, bm25, index_ready = load_index()
    
    if not index_ready or store is None:
        st.error("❌ Index not built. Please run `python scripts/build_index.py` first.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Mode selection
        mode = st.selectbox(
            "Retrieval Mode",
            ["rag_hybrid", "rag_dense", "llm_only"],
            help="rag_hybrid = Dense + BM25, rag_dense = Dense only, llm_only = No retrieval"
        )
        
        # Prompt profile
        profile = st.selectbox(
            "Prompt Profile",
            ["strict", "concise", "verbose"],
            help="strict = Only use context, concise = Short answers, verbose = Detailed explanations"
        )
        
        # Feedback toggle
        use_feedback = st.checkbox("Use Feedback Weights", value=True)
        
        # Show feedback weights
        if use_feedback:
            weights = load_feedback_weights()
            if weights:
                st.write(f"📊 Active feedback weights: {len(weights)} chunks")

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Query")
        
        # Query input
        query = st.text_area(
            "Ask a question about Ghana elections or budget:",
            height=100,
            placeholder="e.g., What was the NDC percentage in Greater Accra in 2020?"
        )
        
        # Submit buttons
        col_submit, col_compare = st.columns(2)
        
        with col_submit:
            if st.button("🚀 Run Query", type="primary"):
                if query.strip():
                    with st.spinner("Processing..."):
                        # Ensure store is not None before using
                        if store is not None:
                            result = run_pipeline(
                                query.strip(),
                                store,
                                bm25 if mode != "llm_only" else None,
                                mode=mode,
                                prompt_profile=profile,
                                use_feedback=use_feedback
                            )
                            st.session_state.current_result = result
                            st.session_state.query_history.append({
                                'query': query,
                                'mode': mode,
                                'time': time.time()
                            })
                        else:
                            st.error("Index not loaded.")
                else:
                    st.error("Please enter a query.")
        
        with col_compare:
            if st.button("🆚 Compare RAG vs LLM"):
                if query.strip():
                    with st.spinner("Comparing..."):
                        # Ensure store is not None before using
                        if store is not None:
                            rag_result = run_pipeline(
                                query.strip(),
                                store,
                                bm25,
                                mode="rag_hybrid",
                                prompt_profile=profile,
                                use_feedback=use_feedback
                            )
                            llm_result = run_pipeline(
                                query.strip(),
                                store,
                                None,
                                mode="llm_only",
                                prompt_profile=profile,
                                use_feedback=False
                            )
                            st.session_state.comparison = {
                                'rag': rag_result,
                                'llm': llm_result,
                                'query': query
                            }
                        else:
                            st.error("Index not loaded.")
                else:
                    st.error("Please enter a query.")

    with col2:
        st.header("📊 Status")
        
        # Show feedback message
        if st.session_state.feedback_message:
            st.markdown(f"<div class='success-message'>{st.session_state.feedback_message}</div>", unsafe_allow_html=True)
            st.session_state.feedback_message = None  # Clear after showing
        
        # Index status
        st.success("✅ Index loaded successfully")
        # Ensure store is not None before accessing chunks
        if store is not None:
            st.info(f"📚 {len(store.chunks)} chunks indexed")
        else:
            st.info("📚 Index not available")
        
        # Recent queries
        if st.session_state.query_history:
            st.subheader("🕐 Recent Queries")
            for i, q in enumerate(reversed(st.session_state.query_history[-5:])):
                st.write(f"{i+1}. {q['query'][:50]}... ({q['mode']})")

    # Results section
    if 'current_result' in st.session_state:
        result = st.session_state.current_result
        
        st.header("📄 Results")
        
        # Answer
        st.subheader("💡 Answer")
        if result.get('answer'):
            st.write(result['answer'])
        elif result.get('llm_error'):
            st.error(f"LLM Error: {result['llm_error']}")
        else:
            st.warning("No answer generated.")
        
        # Retrieved chunks
        if result.get('retrieved'):
            st.subheader("🔍 Retrieved Chunks")
            
            formatted_chunks = format_retrieved_chunks(result['retrieved'])
            
            for chunk in formatted_chunks:
                with st.container():
                    st.markdown(f"""
                    <div class='chunk-card'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;'>
                            <strong>{chunk['source_id']}</strong>
                            <span style='font-size: 0.8em; color: #666;'>{chunk['scores_line']}</span>
                        </div>
                        <p style='margin: 5px 0;'>{chunk['text_preview']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Feedback buttons
                    col_up, col_down = st.columns(2)
                    with col_up:
                        if st.button(f"👍 {chunk['source_id']}", key=f"up_{chunk['rank']}"):
                            handle_feedback(chunk['source_id'], "up")
                    with col_down:
                        if st.button(f"👎 {chunk['source_id']}", key=f"down_{chunk['rank']}"):
                            handle_feedback(chunk['source_id'], "down")
        
        # Pipeline info
        with st.expander("🔧 Pipeline Details"):
            st.json({
                'mode': result.get('mode'),
                'total_time_ms': result.get('total_ms'),
                'stages': result.get('stages', [])
            })

    # Comparison section
    if 'comparison' in st.session_state:
        comp = st.session_state.comparison
        
        st.header("🆚 RAG vs LLM Comparison")
        
        col_rag, col_llm = st.columns(2)
        
        with col_rag:
            st.subheader("🔍 RAG Answer")
            st.write(comp['rag'].get('answer', 'No answer'))
            st.info(f"Retrieved {len(comp['rag'].get('retrieved', []))} chunks")
        
        with col_llm:
            st.subheader("🤖 Pure LLM Answer")
            st.write(comp['llm'].get('answer', 'No answer'))
            st.info("No retrieval used")

    # Manual experiment logs
    st.header("📝 Manual Experiment Logs")
    
    with st.expander("Add Log Entry"):
        log_entry = st.text_area("Log Entry", height=100)
        if st.button("Add Log"):
            if log_entry.strip():
                ensure_dirs()
                with MANUAL_LOGS_PATH.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"ts": time.time(), "entry": log_entry.strip()}, ensure_ascii=False) + "\n")
                st.success("Log entry added!")
    
    # Show existing logs
    if MANUAL_LOGS_PATH.exists():
        with st.expander("View Logs"):
            try:
                with MANUAL_LOGS_PATH.open("r", encoding="utf-8") as f:
                    logs = [json.loads(line) for line in f if line.strip()]
                    for log in reversed(logs[-10:]):  # Show last 10 logs
                        st.write(f"• {log['entry']}")
            except Exception as e:
                st.error(f"Error reading logs: {e}")

if __name__ == "__main__":
    main()
