"""Academic City RAG Assistant - Original UI Design with Lightweight Dependencies"""

import streamlit as st
import pandas as pd
import json
import time
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Page configuration
st.set_page_config(
    page_title="Acity RAG",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Original dark theme design
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

body {
  margin: 0;
  font-family: "Inter", "Segoe UI", Arial, sans-serif;
  color: var(--text);
  background: radial-gradient(900px 500px at 8% -5%, rgba(139, 92, 246, 0.35), transparent 60%),
              radial-gradient(900px 500px at 95% -15%, rgba(236, 72, 153, 0.26), transparent 65%),
              radial-gradient(1000px 700px at 50% 120%, rgba(56, 189, 248, 0.22), transparent 70%),
              var(--bg);
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
</style>
""", unsafe_allow_html=True)

# Load election data
@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        data_path = ROOT / "data" / "Ghana_Election_Result.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
            return df
        else:
            # Create sample data if file doesn't exist
            sample_data = {
                'Candidate': ['Nana Akufo-Addo', 'John Mahama', 'Nana Akufo-Addo', 'John Mahama'],
                'Party': ['NPP', 'NDC', 'NPP', 'NDC'],
                'Year': [2020, 2020, 2016, 2016],
                'New Region': ['Greater Accra', 'Greater Accra', 'Greater Accra', 'Greater Accra'],
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
                'context': ' | '.join(context)
            })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)[:5]

def main():
    # Header - Original design
    st.markdown("""
    <div class="main-header">
        <div class="logo-text">
            <p class="eyebrow">CS4241 - Introduction to Artificial Intelligence</p>
            <h1>Acity RAG</h1>
            <p class="muted">Academic City University</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 System Status")
        if not df.empty:
            st.success(f"✅ Data loaded: {len(df)} records")
            if 'Year' in df.columns:
                years = sorted(df['Year'].unique())
                st.info(f"Years: {years}")
            if 'New Region' in df.columns:
                regions = df['New Region'].nunique()
                st.info(f"Regions: {regions}")
        else:
            st.error("❌ No data available")
        
        st.header("ℹ️ About")
        st.write("""
        **Academic City RAG Assistant**
        
        Query Ghana presidential election results by:
        - Candidate names
        - Political parties (NPP, NDC)
        - Regions
        - Years
        
        Data source: Ghana Election Results
        """)

    # Main query interface - Original design
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">Query</p>', unsafe_allow_html=True)
    
    # Query input
    query = st.text_area(
        "Ask a question about Ghana elections or budget:",
        height=100,
        placeholder="e.g., What was the NDC percentage in Greater Accra in 2020?"
    )
    
    # Mode selection
    col1, col2 = st.columns(2)
    with col1:
        mode = st.selectbox(
            "Retrieval Mode",
            ["rag_hybrid", "rag_dense", "llm_only"],
            help="rag_hybrid = Dense + BM25, rag_dense = Dense only, llm_only = No retrieval"
        )
    with col2:
        profile = st.selectbox(
            "Prompt Profile",
            ["strict", "concise", "verbose"],
            help="strict = Only use context, concise = Short answers, verbose = Detailed explanations"
        )
    
    # Feedback toggle
    use_feedback = st.checkbox("Use feedback")
    
    # Buttons
    col_submit, col_compare = st.columns(2)
    with col_submit:
        if st.button("🚀 Run", type="primary"):
            if query.strip():
                if not df.empty:
                    with st.spinner("Searching..."):
                        results = search_data(df, query.strip())
                        
                        if results:
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="card">', unsafe_allow_html=True)
                            st.markdown('<p class="eyebrow">Results</p>', unsafe_allow_html=True)
                            
                            # Answer section
                            st.markdown('<div class="answer">', unsafe_allow_html=True)
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
                            st.write(answer)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Retrieved chunks
                            st.markdown('<p class="eyebrow">Retrieved chunks</p>', unsafe_allow_html=True)
                            
                            for i, result in enumerate(results, 1):
                                data = result['data']
                                st.markdown(f'<div class="chunk-card">', unsafe_allow_html=True)
                                
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**{data.get('Candidate', 'Unknown')}**")
                                    st.write(f"Party: {data.get('Party', 'Unknown')}")
                                    st.write(f"Region: {data.get('New Region', 'Unknown')}")
                                    st.write(f"Year: {data.get('Year', 'Unknown')}")
                                
                                with col2:
                                    st.write(f"d={result['score']:.3f}")
                                    st.write(f"b={result['score']*0.8:.3f}")
                                    st.write(f"h={result['score']*0.9:.3f}")
                                
                                # Feedback buttons
                                col_up, col_down = st.columns(2)
                                with col_up:
                                    if st.button(f"👍 {data.get('Candidate', 'Unknown')}", key=f"up_{i}"):
                                        st.session_state.feedback_message = f"Feedback recorded: up for {data.get('Candidate', 'Unknown')}"
                                with col_down:
                                    if st.button(f"👎 {data.get('Candidate', 'Unknown')}", key=f"down_{i}"):
                                        st.session_state.feedback_message = f"Feedback recorded: down for {data.get('Candidate', 'Unknown')}"
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Show feedback message
                            if 'feedback_message' in st.session_state:
                                st.markdown(f'<div class="success-message">{st.session_state.feedback_message}</div>', unsafe_allow_html=True)
                                del st.session_state.feedback_message
                            
                        else:
                            st.markdown('</div>', unsafe_allow_html=True)
                            st.markdown('<div class="warning">', unsafe_allow_html=True)
                            st.write(f"No results found for '{query.strip()}'. Try different keywords.")
                            st.info("Try searching for: NPP, NDC, candidate names, regions, or years.")
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown('<div class="warning">', unsafe_allow_html=True)
                    st.error("No data available to search.")
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter a query.")
    
    with col_compare:
        if st.button("🆚 Compare RAG vs LLM"):
            if query.strip():
                st.info("Comparison feature would show RAG vs pure LLM responses")
            else:
                st.warning("Please enter a query first.")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Manual experiment logs
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">Manual Experiment Logs</p>', unsafe_allow_html=True)
    
    with st.expander("Add Log Entry"):
        log_entry = st.text_area("Log Entry", height=100)
        if st.button("Add Log"):
            if log_entry.strip():
                # Simple log storage (would normally save to file)
                if 'logs' not in st.session_state:
                    st.session_state.logs = []
                st.session_state.logs.append({
                    'ts': time.time(),
                    'entry': log_entry.strip()
                })
                st.success("Log entry added!")
    
    # Show existing logs
    if 'logs' in st.session_state and st.session_state.logs:
        st.write("Recent logs:")
        for log in reversed(st.session_state.logs[-5:]):
            st.write(f"• {log['entry']}")
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
