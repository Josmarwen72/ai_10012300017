"""Simple Streamlit RAG App - Lightweight Deployment Version"""

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
</style>
""", unsafe_allow_html=True)

def load_election_data():
    """Load Ghana election data from CSV"""
    try:
        csv_path = ROOT / "data" / "Ghana_Election_Result.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading election data: {e}")
        return None

def search_election_data(df, query):
    """Simple keyword search in election data"""
    if df is None:
        return []
    
    query_lower = query.lower()
    results = []
    
    # Search in relevant columns
    for idx, row in df.iterrows():
        score = 0
        context_parts = []
        
        # Check candidate name
        if 'candidate' in query_lower and pd.notna(row.get('Candidate')):
            candidate = str(row['Candidate']).lower()
            if query_lower in candidate or candidate in query_lower:
                score += 3
                context_parts.append(f"Candidate: {row['Candidate']}")
        
        # Check party
        if any(party in query_lower for party in ['ndc', 'npp', 'cpp', 'party']):
            if pd.notna(row.get('Party')):
                party = str(row['Party']).lower()
                if party in query_lower or any(p in party for p in ['ndc', 'npp', 'cpp']):
                    score += 2
                    context_parts.append(f"Party: {row['Party']}")
        
        # Check region
        if pd.notna(row.get('New Region')):
            region = str(row['New Region']).lower()
            if region in query_lower:
                score += 2
                context_parts.append(f"Region: {row['New Region']}")
        
        # Check year
        if any(year in query_lower for year in ['2020', '2016', '2012', '2008', '2004']):
            if pd.notna(row.get('Year')):
                year = str(row['Year'])
                if year in query_lower:
                    score += 2
                    context_parts.append(f"Year: {row['Year']}")
        
        # Check votes percentage
        if '%' in query_lower or 'percent' in query_lower or 'vote' in query_lower:
            if pd.notna(row.get('Votes(%)')):
                votes_pct = str(row['Votes(%)'])
                score += 1
                context_parts.append(f"Votes: {votes_pct}%")
        
        if score > 0:
            results.append({
                'score': score,
                'data': row.to_dict(),
                'context': ' | '.join(context_parts)
            })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:5]  # Return top 5 results

def generate_response(query, search_results):
    """Generate a simple response based on search results"""
    if not search_results:
        return "I couldn't find specific information about your query in the Ghana election data. Please try more specific terms like candidate names, parties, regions, or years."
    
    response_parts = [f"Based on the Ghana election data, here's what I found about '{query}':\n"]
    
    for i, result in enumerate(search_results[:3], 1):
        row = result['data']
        response_parts.append(f"\n{i}. {row.get('Candidate', 'Unknown')} from {row.get('Party', 'Unknown')} party")
        
        if pd.notna(row.get('New Region')):
            response_parts.append(f"   - Region: {row['New Region']}")
        
        if pd.notna(row.get('Year')):
            response_parts.append(f"   - Year: {row['Year']}")
        
        if pd.notna(row.get('Votes(%)')):
            response_parts.append(f"   - Vote Percentage: {row['Votes(%)']}")
        
        if pd.notna(row.get('Votes')):
            response_parts.append(f"   - Total Votes: {row['Votes']}")
    
    return '\n'.join(response_parts)

def main():
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
        <h1>🎓 Academic City RAG Assistant</h1>
        <p>CS4241 - Introduction to Artificial Intelligence</p>
        <p><em>Ghana Election Data Query System</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    df = load_election_data()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Data Status")
        if df is not None:
            st.success(f"✅ Election data loaded ({len(df)} rows)")
            st.info(f"Years: {sorted(df['Year'].unique())}")
            st.info(f"Regions: {len(df['New Region'].unique())}")
        else:
            st.error("❌ Election data not found")
        
        st.header("ℹ️ About")
        st.write("""
        This is a simplified RAG system for Ghana presidential election results.
        
        **Features:**
        - Search election data by candidate, party, region, year
        - View vote percentages and totals
        - Compare results across regions and years
        
        **Data Source:** Ghana Election Results CSV
        """)

    # Main content
    st.header("💬 Query Ghana Election Data")
    
    # Query input
    query = st.text_input(
        "Ask about Ghana elections:",
        placeholder="e.g., NDC results in Greater Accra 2020, NPP performance, Mahama votes",
        help="Try searching for candidate names, parties (NDC, NPP), regions, or years"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if query.strip():
            if df is not None:
                with st.spinner("Searching election data..."):
                    search_results = search_election_data(df, query.strip())
                    response = generate_response(query.strip(), search_results)
                    
                    # Display response
                    st.subheader("💡 Answer")
                    st.write(response)
                    
                    # Display raw results
                    if search_results:
                        st.subheader("📋 Raw Data Results")
                        for i, result in enumerate(search_results, 1):
                            with st.expander(f"Result {i} (Score: {result['score']})"):
                                st.json(result['data'])
            else:
                st.error("Election data not available. Please ensure the CSV file is in the data folder.")
        else:
            st.error("Please enter a query.")

    # Sample queries section
    st.header("🎯 Sample Queries")
    sample_queries = [
        "NDC results Greater Accra 2020",
        "NPP performance Ashanti region",
        "Mahama vote percentages 2020",
        "Akufo Addo Central Region 2016",
        "Presidential election results Volta Region"
    ]
    
    for sample_query in sample_queries:
        if st.button(sample_query, key=f"sample_{sample_query}"):
            st.session_state.query = sample_query
            st.rerun()

    # Data overview
    if df is not None:
        st.header("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df))
        
        with col2:
            unique_years = df['Year'].nunique()
            st.metric("Years Covered", unique_years)
        
        with col3:
            unique_regions = df['New Region'].nunique()
            st.metric("Regions", unique_regions)
        
        # Show sample data
        with st.expander("📄 Sample Data (First 10 rows)"):
            st.dataframe(df.head(10))

if __name__ == "__main__":
    main()
