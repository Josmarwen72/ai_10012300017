"""Application configuration (no framework RAG — paths and hyperparameters only)."""

from __future__ import annotations

import os
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, will use system environment variables only
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ACITY_DATA_DIR", ROOT / "data"))
INDEX_DIR = DATA_DIR / "index"
LOGS_DIR = DATA_DIR / "logs"

# Embedding model (Sentence-Transformers)
EMBED_MODEL_NAME = os.environ.get(
    "ACITY_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# Chunking defaults (justified in docs/PROJECT_DOCUMENTATION.md)
CSV_CHUNK_MAX_ROWS = 12
PDF_CHUNK_CHARS = 900
PDF_CHUNK_OVERLAP = 150

# Retrieval
TOP_K = int(os.environ.get("ACITY_TOP_K", "8"))
HYBRID_ALPHA = float(os.environ.get("ACITY_HYBRID_ALPHA", "0.65"))  # weight on dense similarity

# Context assembly for LLM
MAX_CONTEXT_CHARS = int(os.environ.get("ACITY_MAX_CONTEXT_CHARS", "6000"))

# Hosted LLM providers (optional). If none is configured, local fallback is used.
LLM_PROVIDER = os.environ.get("ACITY_LLM_PROVIDER", "auto").lower()  # auto | openai | groq | local
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("ACITY_OPENAI_MODEL", "gpt-4o-mini")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("ACITY_GROQ_MODEL", "llama-3.1-8b-instant")

# Local generation fallback (used when OPENAI_API_KEY is not set)
# Note: small instruction-tuned model so it can run on CPU for coursework demos.
LOCAL_LLM_MODEL = os.environ.get("ACITY_LOCAL_LLM_MODEL", "google/flan-t5-base")
LOCAL_LLM_MAX_NEW_TOKENS = int(os.environ.get("ACITY_LOCAL_LLM_MAX_NEW_TOKENS", "220"))

MANUAL_LOGS_PATH = DATA_DIR / "manual_experiment_logs.jsonl"
PIPELINE_LOGS_PATH = DATA_DIR / "pipeline_runs.jsonl"
FEEDBACK_PATH = DATA_DIR / "chunk_feedback.jsonl"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
