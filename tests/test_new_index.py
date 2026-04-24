#!/usr/bin/env python3
"""Test script to directly test the new index with CSV chunks."""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    from backend.vector_store import FaissVectorStore
    from backend.embeddings import embed_query
    from backend.retrieval import retrieve_dense_only
    from backend.bm25 import BM25Index, tokenize
    
    print("🔍 Testing New Index Directly...")
    
    # Load index directly from data/index
    index_dir = Path('data/index')
    if not index_dir.exists():
        print("❌ Index directory not found")
        sys.exit(1)
    
    store = FaissVectorStore.load(index_dir)
    print(f"✅ Index loaded with {len(store.chunks)} chunks")
    
    # Build BM25 index
    docs = [tokenize(c.text) for c in store.chunks]
    bm25 = BM25Index(docs)
    print("✅ BM25 index built")
    
    csv_chunks = [c for c in store.chunks if c.meta.get('type') == 'election_csv']
    pdf_chunks = [c for c in store.chunks if c.meta.get('type') == 'budget_pdf']
    print(f"📊 CSV chunks: {len(csv_chunks)}")
    print(f"📊 PDF chunks: {len(pdf_chunks)}")
    
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
        retrieved = retrieve_dense_only(store, qvec, k=5)
        
        print(f"📊 Retrieved {len(retrieved)} chunks:")
        csv_count = 0
        pdf_count = 0
        for i, rc in enumerate(retrieved):
            chunk_type = rc.chunk.meta.get('type', 'unknown')
            source_id = rc.chunk.source_id
            score = rc.dense_score
            text_preview = rc.chunk.text[:100] + "..."
            print(f"  {i+1}. {chunk_type} (score: {score:.3f}) - {source_id}")
            print(f"     {text_preview}")
            
            if chunk_type == 'election_csv':
                csv_count += 1
            elif chunk_type == 'budget_pdf':
                pdf_count += 1
        
        print(f"📈 Results: {csv_count} CSV chunks, {pdf_count} PDF chunks")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
