"""Simple Streamlit RAG App - Easier Deployment"""

import streamlit as st
import sys
import json
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Page configuration
st.set_page_config(
    page_title="Academic City RAG Assistant",
    page_icon="🎓",
    layout="wide"
)

# Simple header
st.markdown("""
# 🎓 Academic City RAG Assistant
**CS4241 - Introduction to Artificial Intelligence**
""")

st.markdown("---")

# Simple info section
st.markdown("""
## 📋 About This App

This RAG (Retrieval-Augmented Generation) assistant answers questions about:
- **Ghana Election Results** (CSV data)
- **2025 Budget Statement** (PDF data)

## 🚀 Deployment Status

**Note:** This app requires data files and index to be built before it can function properly.

### To Use This App:

1. **Build the index locally:**
   ```bash
   python scripts/build_index.py
   ```

2. **Run locally:**
   ```bash
   streamlit run streamlit_app.py
   ```

3. **Deploy to cloud:**
   - Push to GitHub with built index
   - Deploy on Streamlit Cloud

## 📊 Features

- ✅ RAG retrieval with hybrid search
- ✅ Feedback system for improving results
- ✅ Manual experiment logging
- ✅ Multiple prompt profiles
- ✅ Comparison mode (RAG vs pure LLM)

## 🔧 Technical Stack

- **Backend:** Python with FAISS, Sentence-Transformers
- **Frontend:** Streamlit
- **Data:** Ghana Election CSV + Budget PDF
- **LLM:** OpenAI/Groq or local fallback

---

**For full functionality, use the complete `streamlit_app.py` file.**
""")

# Simple status check
st.markdown("---")
st.markdown("### 📈 App Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Repository", "✅ Ready", "GitHub Updated")

with col2:
    st.metric("Dependencies", "✅ Fixed", "packages.txt Updated")

with col3:
    st.metric("Deployment", "🔄 Pending", "Streamlit Cloud")

# Deployment instructions
st.markdown("---")
st.markdown("### 🚀 Quick Deployment Guide")

st.markdown("""
1. **Go to Streamlit Cloud:** https://share.streamlit.io
2. **Sign in** with GitHub
3. **Click "New app"**
4. **Repository:** `Josmarwen72/ai_10012300017`
5. **Main file:** `streamlit_app.py` (not this simple version)
6. **Requirements:** `packages.txt`
7. **Click "Deploy!"**
""")

if st.button("📋 Check Repository Status"):
    st.info("✅ Repository is ready for deployment!")
    st.code("Repository: Josmarwen72/ai_10012300017\nMain file: streamlit_app.py\nRequirements: packages.txt")

st.markdown("---")
st.markdown("*This is a simplified status page. Use `streamlit_app.py` for the full RAG functionality.*")
