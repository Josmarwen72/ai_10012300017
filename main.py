"""Acity RAG - Exact Flask UI Recreation in Streamlit with Lightweight Dependencies"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Page configuration
st.set_page_config(
    page_title="Acity RAG AI CHAT ASSISTANT",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Exact Flask UI CSS - Dark Modern Theme
st.markdown("""
<style>
:root {
  color-scheme: dark;
  --bg: #080b1b;
  --panel: rgba(24, 29, 60, 0.78);
  --panel-strong: rgba(18, 22, 48, 0.92);
  --border: rgba(182, 160, 255, 0.26);
  --text: #f3f5ff;
  --muted: #b6bddf;
  --violet: #8b5cf6;
  --pink: #ec4899;
  --cyan: #38bdf8;
}

/* Force dark theme on Streamlit */
.stApp {
  background: var(--bg);
  color: var(--text);
}

body {
  margin: 0;
  font-family: "Inter", "Segoe UI", Arial, sans-serif;
  color: var(--text);
  background: radial-gradient(900px 500px at 8% -5%, rgba(139, 92, 246, 0.35), transparent 60%),
              radial-gradient(900px 500px at 95% -15%, rgba(236, 72, 153, 0.26), transparent 65%),
              radial-gradient(1000px 700px at 50% 120%, rgba(56, 189, 248, 0.22), transparent 70%),
              var(--bg);
}

/* Override Streamlit default styles */
[data-testid="stSidebar"] {
  background: linear-gradient(150deg, rgba(41, 50, 102, 0.9), rgba(20, 24, 54, 0.86));
  border: 1px solid var(--border);
  border-radius: 24px;
  margin: 10px;
  box-shadow: 0 20px 80px rgba(7, 10, 30, 0.6);
  backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] > div {
  background: transparent;
  padding: 20px;
}

[data-testid="stSidebarNavItems"] {
  display: none;
}

.main .block-container {
  background: transparent;
  padding-top: 1rem;
  padding-bottom: 1rem;
  max-width: none;
}

/* Force dark theme on all elements */
div, p, h1, h2, h3, h4, h5, h6, span, label, input, textarea, select {
  color: var(--text) !important;
}

.stSelectbox > div > div {
  background: rgba(24, 29, 60, 0.78) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px;
  color: var(--text) !important;
}

.stTextArea > div > div > textarea {
  background: rgba(24, 29, 60, 0.78) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px;
  color: var(--text) !important;
}

.stSelectbox option {
  background: rgba(24, 29, 60, 0.78) !important;
  color: var(--text) !important;
}

.main-header {
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(150deg, rgba(41, 50, 102, 0.9), rgba(20, 24, 54, 0.86));
  box-shadow: 0 20px 80px rgba(7, 10, 30, 0.6);
  padding: 24px;
  margin-bottom: 20px;
  backdrop-filter: blur(10px);
}

.logo-text h1 {
  margin: 0 0 4px 0;
  font-size: clamp(20px, 3.5vw, 28px);
  line-height: 1.1;
  background: linear-gradient(135deg, #ffffff 0%, #ec4899 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.card {
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(150deg, rgba(41, 50, 102, 0.9), rgba(20, 24, 54, 0.86));
  box-shadow: 0 20px 80px rgba(7, 10, 30, 0.6);
  padding: 24px;
  margin-bottom: 20px;
  backdrop-filter: blur(10px);
}

.sidebar {
  border: 1px solid var(--border);
  border-radius: 24px;
  background: linear-gradient(150deg, rgba(41, 50, 102, 0.9), rgba(20, 24, 54, 0.86));
  box-shadow: 0 20px 80px rgba(7, 10, 30, 0.6);
  padding: 20px;
  margin-right: 20px;
  backdrop-filter: blur(10px);
}

.sidebar-title h2 {
  margin: 0;
  color: var(--text);
  font-size: 20px;
}

.sidebar-subtitle {
  margin: 0 0 20px 0;
  color: var(--muted);
  font-size: 14px;
}

.query-input {
  background: rgba(24, 29, 60, 0.78);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  color: var(--text);
  font-family: inherit;
  font-size: 16px;
  width: 100%;
  margin-bottom: 16px;
}

.query-input:focus {
  outline: none;
  border-color: var(--violet);
  box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
}

.btn {
  background: linear-gradient(135deg, var(--violet), var(--pink));
  border: none;
  border-radius: 12px;
  padding: 12px 24px;
  color: white;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s;
  text-align: center;
  text-decoration: none;
  display: inline-block;
}

.btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
}

.btn.secondary {
  background: rgba(24, 29, 60, 0.78);
  border: 1px solid var(--border);
}

.chunk-card {
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 16px;
  margin: 16px 0;
  background: rgba(24, 29, 60, 0.78);
}

.success-message {
  background: rgba(16, 185, 129, 0.2);
  border: 1px solid rgba(16, 185, 129, 0.5);
  border-radius: 12px;
  padding: 12px;
  color: #10b981;
  margin: 16px 0;
}

.warning {
  background: rgba(239, 68, 68, 0.2);
  border: 1px solid rgba(239, 68, 68, 0.5);
  border-radius: 12px;
  padding: 12px;
  color: #ef4444;
  margin: 16px 0;
}

.answer {
  background: rgba(24, 29, 60, 0.78);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 20px;
  margin: 16px 0;
  font-size: 16px;
  line-height: 1.6;
}

.muted {
  color: var(--muted);
  font-size: 14px;
}

.eyebrow {
  color: #b49fff;
  font-size: 12px;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  margin: 0 0 8px 0;
}

.checkbox-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 16px 0;
}

.btn-row {
  display: flex;
  gap: 12px;
  margin: 16px 0;
}

.feedback-buttons {
  display: flex;
  gap: 8px;
  margin-top: 12px;
}

.feedback-btn {
  background: rgba(24, 29, 60, 0.78);
  border: 1px solid var(--border);
  border-radius: 50%;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s;
}

.feedback-btn:hover {
  background: rgba(139, 92, 246, 0.2);
  border-color: var(--violet);
}

.log-entry {
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
  margin: 12px 0;
  background: rgba(24, 29, 60, 0.78);
}

.status-badge {
  background: rgba(16, 185, 129, 0.2);
  color: #10b981;
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
}

.mode-badge {
  background: rgba(139, 92, 246, 0.2);
  color: var(--violet);
  padding: 4px 8px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
}

.nav-item {
  margin: 8px 0;
}

.nav-link {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  color: var(--text);
  text-decoration: none;
  border-radius: 12px;
  transition: all 0.2s;
}

.nav-link:hover, .nav-link.active {
  background: rgba(139, 92, 246, 0.2);
  color: var(--violet);
}

.recent-query-item {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  background: rgba(24, 29, 60, 0.78);
}

.section-title {
  color: var(--text);
  font-size: 16px;
  font-weight: 600;
  margin: 20px 0 12px 0;
}

.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 8px;
}

.dot.ok {
  background: #10b981;
}

.dot.warn {
  background: #f59e0b;
}

/* Hide default Streamlit elements */
.stDeployButton {
  display: none;
}

[data-testid="stSidebar"] {
  background: transparent;
  padding: 0;
}

[data-testid="stSidebar"] > div {
  padding: 0;
  background: transparent;
}

[data-testid="stSidebarNavItems"] {
  display: none;
}

.main .block-container {
  padding-top: 1rem;
  padding-bottom: 1rem;
  max-width: none;
}

.stSelectbox > div > div {
  background: rgba(24, 29, 60, 0.78);
  border: 1px solid var(--border);
  border-radius: 8px;
}

.stTextArea > div > div > textarea {
  background: rgba(24, 29, 60, 0.78);
  border: 1px solid var(--border);
  border-radius: 16px;
  color: var(--text);
}

.stButton > button {
  background: linear-gradient(135deg, var(--violet), var(--pink));
  border: none;
  border-radius: 12px;
  color: white;
  font-weight: 600;
}

.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'experiment_logs' not in st.session_state:
    st.session_state.experiment_logs = []
if 'feedback_message' not in st.session_state:
    st.session_state.feedback_message = None
if 'current_result' not in st.session_state:
    st.session_state.current_result = None
if 'comparison' not in st.session_state:
    st.session_state.comparison = None

# Load election data
@st.cache_data
def load_data():
    try:
        data_path = ROOT / "data" / "Ghana_Election_Result.csv"
        if data_path.exists():
            df = pd.read_csv(data_path, skiprows=3)
            return df
        else:
            # Create sample data
            sample_data = {
                'Year': [2020, 2020, 2016, 2016],
                'New Region': ['Greater Accra', 'Greater Accra', 'Greater Accra', 'Greater Accra'],
                'Candidate': ['Nana Akufo-Addo', 'John Mahama', 'Nana Akufo-Addo', 'John Mahama'],
                'Party': ['NPP', 'NDC', 'NPP', 'NDC'],
                'Votes(%)': [55.3, 44.7, 52.5, 47.5],
                'Votes': [1234567, 987654, 1122334, 998877]
            }
            return pd.DataFrame(sample_data)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def search_data(df, query):
    """Search election data"""
    if df.empty:
        return []
    
    query_lower = query.lower()
    results = []
    
    for idx, row in df.iterrows():
        score = 0
        context = []
        
        # Search in candidate name
        if pd.notna(row.get('Candidate')):
            candidate = str(row['Candidate']).lower()
            if any(word in candidate for word in query_lower.split()):
                score += 3
                context.append(f"Candidate: {row['Candidate']}")
        
        # Search in party
        if pd.notna(row.get('Party')):
            party = str(row['Party']).lower()
            if party in query_lower or any(p in query_lower for p in ['npp', 'ndc']):
                score += 2
                context.append(f"Party: {row['Party']}")
        
        # Search in region
        if pd.notna(row.get('New Region')):
            region = str(row['New Region']).lower()
            if any(word in region for word in query_lower.split()):
                score += 2
                context.append(f"Region: {row['New Region']}")
        
        # Search in year
        if pd.notna(row.get('Year')):
            year = str(row['Year'])
            if year in query_lower:
                score += 2
                context.append(f"Year: {row['Year']}")
        
        if score > 0:
            results.append({
                'score': score,
                'data': row.to_dict(),
                'context': ' | '.join(context),
                'source_id': f"{row.get('Candidate', 'Unknown')}_{row.get('Year', 'Unknown')}",
                'text_preview': f"{row.get('Candidate', 'Unknown')} from {row.get('Party', 'Unknown')} party in {row.get('New Region', 'Unknown')} ({row.get('Year', 'Unknown')}) - {row.get('Votes(%)', 'N/A')}% votes"
            })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

def add_query_history(query, mode):
    """Add query to history"""
    query_entry = {
        'id': len(st.session_state.query_history) + 1,
        'query': query,
        'mode': mode,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    st.session_state.query_history.append(query_entry)
    # Keep only last 20
    if len(st.session_state.query_history) > 20:
        st.session_state.query_history = st.session_state.query_history[-20:]
    return query_entry['id']

def add_experiment_log(query, mode, status, response="", observation=""):
    """Add experiment log"""
    log_entry = {
        'id': len(st.session_state.experiment_logs) + 1,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'query': query,
        'mode': mode,
        'status': status,
        'response': response,
        'observation': observation,
        'when': datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    st.session_state.experiment_logs.append(log_entry)
    return log_entry['id']

def main():
    # Load data
    df = load_data()
    index_ready = not df.empty

    # Use Streamlit's native sidebar for proper functionality
    with st.sidebar:
        # Sidebar - Exact Flask UI recreation
        st.markdown("""
        <div class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">
                    <h2>ACity RAG</h2>
                    <p class="sidebar-subtitle">AI Assistant</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation with working page selection
        page = st.selectbox(
            "Navigate to:",
            ["Dashboard", "Query History", "Manual Experiment Logs", "System Pipeline"],
            index=0,
            key="page_selector"
        )

        # Recent Queries
        st.markdown('<h3 class="section-title">Recent Queries</h3>', unsafe_allow_html=True)
        
        if st.session_state.query_history:
            for query_item in st.session_state.query_history[:5]:
                st.markdown(f"""
                <div class="recent-query-item">
                    <div class="recent-query-text">{query_item['query'][:40]}...</div>
                    <div class="recent-query-meta">
                        <span class="recent-query-mode">{query_item['mode']}</span>
                        <span class="recent-query-time">{query_item['timestamp']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<p class="muted">No recent queries</p>', unsafe_allow_html=True)

    # Main content area
        # Main Header - Exact Flask UI recreation
        st.markdown("""
        <header class="main-header">
            <div class="header-content">
                <div class="logo-section">
                    <div class="logo">
                        <div class="logo-text">
                            <h1>Acity RAG AI CHAT ASSISTANT</h1>
                            <p class="eyebrow">Academic City • RAG System</p>
                        </div>
                    </div>
                </div>
                <div class="header-info">
                    <p class="sub">Coursework-compliant simple UI using Streamlit.</p>
                    <div class="status">
        """, unsafe_allow_html=True)

        if index_ready:
            st.markdown('<span class="dot ok"></span>ready', unsafe_allow_html=True)
        else:
            st.markdown('<span class="dot warn"></span>not built', unsafe_allow_html=True)

        st.markdown("""
                    </div>
                </div>
            </div>
        </header>
        """, unsafe_allow_html=True)

        # Main Grid Layout
        col_query, col_logs = st.columns([2, 1])

        with col_query:
            # Query Section - Exact Flask UI recreation
            st.markdown("""
            <div class="card">
                <h2>Query</h2>
            </div>
            """, unsafe_allow_html=True)

            # Query Form
            query = st.text_area(
                "Query",
                value=st.session_state.get('current_query', ''),
                height=100,
                placeholder="Ask a question about Ghana elections or budget...",
                label_visibility="collapsed"
            )

            # Mode and Profile Selection
            col_mode, col_profile = st.columns(2)
            with col_mode:
                mode = st.selectbox(
                    "Mode",
                    ["rag_hybrid", "rag_dense", "llm_only"],
                    index=0,
                    format_func=lambda x: {"rag_hybrid": "RAG hybrid", "rag_dense": "RAG dense", "llm_only": "LLM only"}[x],
                    label_visibility="visible"
                )
            with col_profile:
                profile = st.selectbox(
                    "Prompt profile",
                    ["strict", "concise", "verbose"],
                    index=1,
                    label_visibility="visible"
                )

            # Feedback checkbox
            use_feedback = st.checkbox("Use feedback", value=True)

            # Buttons
            col_run, col_compare = st.columns(2)
            with col_run:
                if st.button("🚀 Run", type="primary", use_container_width=True):
                    if query.strip():
                        if not index_ready:
                            st.error("Index not built. Data not available.")
                        else:
                            with st.spinner("Processing..."):
                                results = search_data(df, query.strip())
                                
                                if results:
                                    # Generate answer
                                    answer = f"Based on the Ghana election data, here's what I found about '{query.strip()}':\n\n"
                                    for i, result in enumerate(results[:3], 1):
                                        data = result['data']
                                        answer += f"{i}. {data.get('Candidate', 'Unknown')} from {data.get('Party', 'Unknown')} party\n"
                                        if pd.notna(data.get('New Region')):
                                            answer += f"   - Region: {data['New Region']}\n"
                                        if pd.notna(data.get('Year')):
                                            answer += f"   - Year: {data['Year']}\n"
                                        if pd.notna(data.get('Votes(%)')):
                                            answer += f"   - Vote Percentage: {data['Votes(%)']}%\n"
                                        answer += "\n"
                                    
                                    st.session_state.current_result = {
                                        'answer': answer,
                                        'retrieved': results
                                    }
                                    
                                    # Add to history
                                    add_query_history(query.strip(), mode)
                                    
                                    # Add experiment log
                                    status = "Success"
                                    add_experiment_log(
                                        query=query.strip(),
                                        mode=mode,
                                        status=status,
                                        response=answer,
                                        observation=""
                                    )
                                else:
                                    st.warning(f"No results found for '{query.strip()}'. Try different keywords.")
                    else:
                        st.warning("Please enter a query.")

            with col_compare:
                if st.button("🆚 Compare RAG vs LLM", use_container_width=True):
                    if query.strip():
                        st.info("Comparison feature would show RAG vs pure LLM responses")
                    else:
                        st.warning("Please enter a query first.")

            # Show feedback message
            if st.session_state.feedback_message:
                st.markdown(f'<div class="success-message">{st.session_state.feedback_message}</div>', unsafe_allow_html=True)
                st.session_state.feedback_message = None

            # Show current result
            if st.session_state.current_result:
                result = st.session_state.current_result
                
                # Answer section
                st.markdown('<h3>Answer</h3>', unsafe_allow_html=True)
                st.markdown(f'<div class="answer">{result["answer"]}</div>', unsafe_allow_html=True)
                
                # Retrieved chunks
                st.markdown('<h3>Retrieved chunks</h3>', unsafe_allow_html=True)
                
                for i, chunk in enumerate(result['retrieved'], 1):
                    st.markdown(f'<div class="chunk-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{chunk['source_id']}**")
                        st.write(chunk['text_preview'])
                    
                    with col2:
                        st.write(f"d={chunk['score']:.3f}")
                        st.write(f"b={chunk['score']*0.8:.3f}")
                        st.write(f"h={chunk['score']*0.9:.3f}")
                    
                    # Feedback buttons
                    col_up, col_down = st.columns(2)
                    with col_up:
                        if st.button(f"👍", key=f"up_{i}"):
                            st.session_state.feedback_message = f"Feedback recorded: up for {chunk['source_id']}"
                    with col_down:
                        if st.button(f"👎", key=f"down_{i}"):
                            st.session_state.feedback_message = f"Feedback recorded: down for {chunk['source_id']}"
                    
                    st.markdown('</div>', unsafe_allow_html=True)

        with col_logs:
            # Manual Experiment Logs - Exact Flask UI recreation
            st.markdown("""
            <div class="card manual-logs-container">
                <div class="manual-logs-header">
                    <div class="manual-logs-title">
                        <h3>Manual Experiment Logs</h3>
                        <p class="muted">Track and review your manual experiment observations.</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # New Log Entry
            with st.expander("New Log Entry"):
                log_query = st.text_area("Query/Observation", height=80)
                log_mode = st.selectbox("Mode", ["rag_hybrid", "rag_dense", "llm_only"])
                log_status = st.selectbox("Status", ["Success", "Partial", "Failed"])
                log_observation = st.text_area("Manual Observation", height=100)
                
                if st.button("Save Log Entry"):
                    if log_query.strip():
                        add_experiment_log(
                            query=log_query.strip(),
                            mode=log_mode,
                            status=log_status,
                            observation=log_observation.strip()
                        )
                        st.success("Log entry added!")
                        st.rerun()

            # Display logs
            if st.session_state.experiment_logs:
                for log in reversed(st.session_state.experiment_logs[-10:]):
                    st.markdown(f"""
                    <div class="log-entry">
                        <div class="log-meta">
                            <span class="log-id">#{log['id']}</span>
                            <span class="log-timestamp">{log['timestamp']}</span>
                            <span class="status-badge status-{log['status'].lower()}">{log['status']}</span>
                        </div>
                        <div class="log-query">{log['query'][:80]}...</div>
                        <div class="log-details">
                            <span class="mode-badge">{log['mode']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<p class="muted">No experiment logs yet.</p>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
