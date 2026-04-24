#!/usr/bin/env python3
"""Test script to debug retrieval issues with CSV vs PDF data."""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    # Import the Flask app's get_index function
    sys.path.insert(0, str(Path(__file__).parent / "flask_ui"))
    from app import get_index
    from backend.embeddings import embed_query
    from backend.retrieval import retrieve_dense_only
    
    print("🔍 Testing Retrieval System...")
    
    # Load index
    store, bm25 = get_index()
    if store is None:
        print("❌ Index not loaded")
        sys.exit(1)
    
    print(f"✅ Index loaded with {len(store.chunks)} chunks")
    print(f"� BM25 available: {bm25 is not None}")
    
    # Test queries about elections (should retrieve CSV data)
    election_queries = [
        "Who won the Ghana election in 1992?",
        "What were the election results in Ashanti Region?",
        "How many votes did Rawlings get?",
        "Which party won the 1996 election?"
    ]
    
    print("\n🗳️ Testing Election Queries (should retrieve CSV data):")
    for query in election_queries:
        print(f"\n📝 Query: {query}")
        qvec = embed_query(query)
        
        # Test dense retrieval only
        retrieved = retrieve_dense_only(store, qvec, k=3)
        
        print(f"📊 Retrieved {len(retrieved)} chunks:")
        for i, rc in enumerate(retrieved):
            chunk_type = rc.chunk.meta.get('type', 'unknown')
            source_id = rc.chunk.source_id
            score = rc.dense_score
            text_preview = rc.chunk.text[:100] + "..."
            print(f"  {i+1}. {chunk_type} (score: {score:.3f}) - {source_id}")
            print(f"     {text_preview}")
    
    # Test queries about budget (should retrieve PDF data)
    budget_queries = [
        "What is the 2025 budget allocation?",
        "How much is allocated to education?",
        "What are the economic policies for 2025?"
    ]
    
    print("\n💰 Testing Budget Queries (should retrieve PDF data):")
    for query in budget_queries:
        print(f"\n📝 Query: {query}")
        qvec = embed_query(query)
        
        # Test dense retrieval only
        retrieved = retrieve_dense_only(store, qvec, k=3)
        
        print(f"📊 Retrieved {len(retrieved)} chunks:")
        for i, rc in enumerate(retrieved):
            chunk_type = rc.chunk.meta.get('type', 'unknown')
            source_id = rc.chunk.source_id
            score = rc.dense_score
            text_preview = rc.chunk.text[:100] + "..."
            print(f"  {i+1}. {chunk_type} (score: {score:.3f}) - {source_id}")
            print(f"     {text_preview}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
