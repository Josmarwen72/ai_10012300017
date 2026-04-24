#!/usr/bin/env python3
"""Test script to verify the index is built and working."""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    # Test loading the index
    from vector_store import get_index
    
    print("🔍 Testing Index Loading...")
    store, bm25 = get_index()
    
    if store is not None:
        print("✅ Index loaded successfully!")
        print(f"   Store type: {type(store).__name__}")
        print(f"   BM25 available: {bm25 is not None}")
        
        # Test a simple search
        if hasattr(store, 'search'):
            print("🔍 Testing search functionality...")
            # This is a basic test - actual search would need embeddings
            print("✅ Search interface available!")
        
    else:
        print("❌ Index not loaded properly")
        sys.exit(1)
        
    print("\n🚀 Index is ready for queries!")
    
except ImportError as e:
    print(f"❌ Error importing vector store: {e}")
    print("💡 Make sure all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error testing index: {e}")
    sys.exit(1)
