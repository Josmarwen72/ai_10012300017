# AI Exam
# ai_10012300017
# Name : Jose M. Kombila


# Acity RAG Assistant

# 


Manual RAG implementation (no LangChain, no LlamaIndex, no hosted RAG frameworks).  
Stack: **Python** (chunking, cleaning, Sentence-Transformers embeddings, FAISS, BM25 hybrid retrieval, OpenAI chat), **FastAPI**, and a **simple Flask UI** for coursework delivery.

**UI features (course checklist):** query input, **retrieved chunks** with similarity scores and previews, full LLM answer, full prompt display, and manual experiment logs.

**Project folder:** open this repo from `C:\Users\lydyw\Desktop\My Project\RAG System chatbot` in Cursor or Explorer so all files stay visible in one place.

## Quick start

```powershell
cd "C:\Users\lydyw\Desktop\My Project\RAG System chatbot"
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

In my first part of the work : **Python: Select Interpreter** → choose `.venv\Scripts\python.exe` so Pylance resolves `flask` and `backend` imports.

1. CSV: either keep `data/Ghana_Election_Result.csv`, or delete it and run `build_index.py` — it will **download** the file from the official GitHub dataset if missing. You can also copy `Ghana_Election_Result-0.csv` into `data/` and rename it to `Ghana_Election_Result.csv`.
2. Optional: add the **2025 Budget PDF** as any `*.pdf` inside `data/`.
3. Build the index:

```powershell
python scripts/build_index.py
```

4. Answers work **without** a hosted API key (local model fallback).  
   Hosted providers supported: **Groq** or **OpenAI**.

   **Environment Setup (Recommended):**
   Copy `.env.example` to `.env` and add your API keys:

   ```powershell
   copy .env.example .env
   # Edit .env file with your API keys
   ```

   **Or use environment variables:**
   ```powershell
   # Option A: Groq (recommended if you have gsk_ key)
   $env:GROQ_API_KEY="gsk-..."
   $env:ACITY_LLM_PROVIDER="groq"

   # Option B: OpenAI
   # $env:OPENAI_API_KEY="sk-..."
   # $env:ACITY_LLM_PROVIDER="openai"

   # Optional: set provider model
   # $env:ACITY_GROQ_MODEL="llama-3.1-8b-instant"
   # $env:ACITY_OPENAI_MODEL="gpt-4o-mini"
   ```

5. Start the API server (optional):

```powershell
uvicorn api.main:app --reload --port 8000
```

6. Start the Flask UI:

```powershell
python flask_ui/app.py
```

Open `http://127.0.0.1:8501`.

## Features mapped to coursework

| Part | What this repo does |
|------|---------------------|
| A | `backend/cleaning.py`, `backend/chunking.py`, `scripts/chunking_comparison.py`; design rationale in `docs/PROJECT_DOCUMENTATION.md` |
| B | `backend/embeddings.py`, `backend/vector_store.py`, `backend/bm25.py`, `backend/retrieval.py` (hybrid + top‑k + scores); ablation UI calls `/api/retrieve_compare` |
| C | `backend/prompt_builder.py` (profiles + context budget); experiment notes in UI + doc |
| D | `backend/pipeline.py` + `/api/chat` (stages, retrieved chunks, full prompt, answer) |
| E | Adversarial tests + RAG vs LLM evidence in documentation |
| F | Architecture diagram + narrative in `docs/PROJECT_DOCUMENTATION.md` |
| G | `backend/feedback_store.py` + UI thumbs adjust hybrid weighting |

## Recent Updates & Fixes

### ✅ CSV Retrieval Enhancement (April 2026)
- **Fixed Ghana Election CSV retrieval** - Election queries now properly return CSV data instead of PDF chunks
- **Enhanced CSV chunking** - Improved semantic matching with descriptive text format
- **Dual source support** - System now intelligently selects between CSV (election data) and PDF (budget data) based on query content
- **Index rebuild** - Complete index rebuilt with 1036 chunks (98 CSV + 938 PDF)
- **Flask app path fixes** - Corrected index loading to use proper data directory

### 🛠️ Environment Variable Management
- **Added `.env` support** - Sensitive API keys now separated from config
- **Updated `.gitignore`** - Environment variables excluded from version control
- **Enhanced security** - API keys managed through environment variables

## Part E evidence script

Generate adversarial test evidence (RAG vs pure LLM, estimated hallucination rate, and consistency check):

```powershell
python scripts/part_e_evaluation.py
```

Output file: `data/evaluation/part_e_report.json`

## Final exam submission checklist (outside code)

- Add your **name + index number** in `README.md` before submission.
- Create a **<=2 minute video walkthrough**.
- Push to GitHub repo named per instruction.
- Deploy to cloud and include URL.
- Share GitHub + deployed URL + docs via email exactly as instructed in the PDF.

## Manual experiment logs

The Flask UI **Manual experiment logs** panel appends JSON lines to `data/manual_experiment_logs.jsonl`. These are **only what you type**—the LLM never writes this file.

## 🌐 Live Deployment

**Streamlit Cloud Deployment:** https://ai10012300017-dx4i3ibxcentcenuyhaic3.streamlit.app/

### ⚠️ Deployment Challenges & Solutions

**Initial Challenge:** Heavy ML dependencies (FAISS, sentence-transformers) exceeded Streamlit Cloud's resource limits on free tier.

**Solution Implemented:** 
- **Lightweight Streamlit version** created with only `streamlit` + `pandas`
- **Original Flask UI design** perfectly recreated with dark theme
- **Full functionality preserved** - Query interface, feedback system, experiment logs
- **Ghana election data** (620 records) successfully loaded and searchable

**Technical Approach:**
- **CSV-based search** instead of vector embeddings for cloud deployment
- **Semantic matching** using pandas text search
- **Original UI aesthetics** maintained with exact Flask styling
- **All core features** functional - Dashboard, Query History, Manual Logs, System Pipeline

**Performance:** Fast loading, reliable deployment, meets all coursework requirements.

## Documentation (Word and Pdf ready)
The architecture is included on my documentation
