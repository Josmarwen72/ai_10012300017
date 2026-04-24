#!/usr/bin/env python3
"""Test script to verify Flask app is loading the correct index."""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

# Add the Flask app directory to the path
flask_dir = Path(__file__).parent / "flask_ui"
sys.path.insert(0, str(flask_dir))

try:
    # Import Flask app's get_index function
    from app import get_index
    from backend.config import INDEX_DIR, DATA_DIR
    
    print("🔍 Testing Flask App Index Loading...")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"INDEX_DIR: {INDEX_DIR}")
    print(f"INDEX_DIR exists: {INDEX_DIR.exists()}")
    
    # Load index using Flask app's function
    store, bm25 = get_index()
    
    if store is None:
        print("❌ Index not loaded by Flask app")
        sys.exit(1)
    
    print(f"✅ Flask app loaded {len(store.chunks)} chunks")
    
    # Check chunk types
    csv_chunks = [c for c in store.chunks if c.meta.get('type') == 'election_csv']
    pdf_chunks = [c for c in store.chunks if c.meta.get('type') == 'budget_pdf']
    
    print(f"📊 CSV chunks: {len(csv_chunks)}")
    print(f"📊 PDF chunks: {len(pdf_chunks)}")
    
    if len(csv_chunks) > 0:
        print("✅ Flask app is loading the new index with CSV chunks!")
        print("Sample CSV chunk:")
        print(csv_chunks[0].text[:150] + "...")
    else:
        print("❌ Flask app is still loading the old index without CSV chunks")
        
    # Test a simple election query
    from backend.embeddings import embed_query
    from backend.retrieval import retrieve_dense_only
    
    print("\n🗳️ Testing election query in Flask app...")
    query = "How many votes did Rawlings get?"
    qvec = embed_query(query)
    retrieved = retrieve_dense_only(store, qvec, k=3)
    
    csv_count = sum(1 for rc in retrieved if rc.chunk.meta.get('type') == 'election_csv')
    pdf_count = sum(1 for rc in retrieved if rc.chunk.meta.get('type') == 'budget_pdf')
    
    print(f"📊 Retrieved {len(retrieved)} chunks: {csv_count} CSV, {pdf_count} PDF")
    
    if csv_count > 0:
        print("✅ Election query returns CSV chunks in Flask app!")
    else:
        print("❌ Election query still returns only PDF chunks in Flask app")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
