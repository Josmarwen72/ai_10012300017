#!/usr/bin/env python3
"""Test script to verify environment variables are loaded correctly."""

import sys
import os
from pathlib import Path

# Add the backend directory to the path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

try:
    # Import the config to test environment variable loading
    from config import GROQ_API_KEY, OPENAI_API_KEY, LLM_PROVIDER
    
    print("✅ Environment Variables Test Results:")
    print(f"   LLM Provider: {LLM_PROVIDER}")
    print(f"   Groq API Key: {'✅ Set' if GROQ_API_KEY else '❌ Not set'}")
    print(f"   OpenAI API Key: {'✅ Set' if OPENAI_API_KEY else '❌ Not set'}")
    
    if GROQ_API_KEY:
        print(f"   Groq Key Length: {len(GROQ_API_KEY)} characters")
    
    print("\n✅ Config loaded successfully from environment variables!")
    
except ImportError as e:
    print(f"❌ Error importing config: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error testing environment variables: {e}")
    sys.exit(1)
