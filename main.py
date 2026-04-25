"""Academic City RAG Assistant - Working Streamlit App"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Academic City RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

# Load election data
@st.cache_data
def load_data():
    try:
        # Try to load the CSV file
        data_path = Path(__file__).parent / "data" / "Ghana_Election_Result.csv"
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
    # Header
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;'>
        <h1>🎓 Academic City RAG Assistant</h1>
        <p>CS4241 - Introduction to Artificial Intelligence</p>
        <p><em>Ghana Election Data Query System</em></p>
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

    # Main query interface
    st.header("💬 Query Election Data")
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., NPP results 2020, Mahama votes, Greater Accra"
    )
    
    # Search button
    if st.button("🔍 Search", type="primary"):
        if query.strip():
            if not df.empty:
                with st.spinner("Searching..."):
                    results = search_data(df, query.strip())
                    
                    if results:
                        st.subheader("📋 Results")
                        
                        for i, result in enumerate(results, 1):
                            with st.expander(f"Result {i} (Score: {result['score']})"):
                                data = result['data']
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if pd.notna(data.get('Candidate')):
                                        st.write(f"**Candidate:** {data['Candidate']}")
                                    if pd.notna(data.get('Party')):
                                        st.write(f"**Party:** {data['Party']}")
                                    if pd.notna(data.get('Year')):
                                        st.write(f"**Year:** {data['Year']}")
                                
                                with col2:
                                    if pd.notna(data.get('New Region')):
                                        st.write(f"**Region:** {data['New Region']}")
                                    if pd.notna(data.get('Votes(%)')):
                                        st.write(f"**Vote %:** {data['Votes(%)']}")
                                    if pd.notna(data.get('Votes')):
                                        st.write(f"**Votes:** {data['Votes']}")
                        
                        # Summary
                        st.subheader("📄 Summary")
                        summary = f"Found {len(results)} results for '{query.strip()}'. "
                        summary += f"Top result: {results[0]['data'].get('Candidate', 'Unknown')} "
                        summary += f"from {results[0]['data'].get('Party', 'Unknown')} party "
                        summary += f"in {results[0]['data'].get('New Region', 'Unknown')} "
                        summary += f"({results[0]['data'].get('Year', 'Unknown')})."
                        st.write(summary)
                        
                    else:
                        st.warning(f"No results found for '{query.strip()}'. Try different keywords.")
                        st.info("Try searching for: NPP, NDC, candidate names, regions, or years.")
            else:
                st.error("No data available to search.")
        else:
            st.warning("Please enter a query.")

    # Sample queries
    st.header("🎯 Sample Queries")
    
    sample_queries = [
        "NPP 2020 results",
        "NDC Greater Accra",
        "Mahama votes",
        "Akufo Addo percentage",
        "Ashanti Region 2016"
    ]
    
    cols = st.columns(3)
    for i, sample in enumerate(sample_queries):
        col = cols[i % 3]
        if col.button(sample, key=f"sample_{i}"):
            st.session_state.query = sample
            st.rerun()

    # Data overview
    if not df.empty:
        st.header("📊 Data Overview")
        
        # Statistics
        if 'Candidate' in df.columns:
            st.subheader("📈 Candidate Performance")
            candidate_stats = df.groupby('Candidate')['Votes(%)'].mean().sort_values(ascending=False)
            st.bar_chart(candidate_stats)
        
        if 'Year' in df.columns and 'Party' in df.columns:
            st.subheader("📅 Yearly Performance")
            yearly_stats = df.groupby(['Year', 'Party'])['Votes(%)'].mean().unstack()
            st.line_chart(yearly_stats)
        
        # Show raw data
        with st.expander("📄 View Raw Data"):
            st.dataframe(df)

if __name__ == "__main__":
    main()
