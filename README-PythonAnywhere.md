# PythonAnywhere Deployment

## Setup for Flask RAG App

### 1. Create Account
- Go to https://www.pythonanywhere.com
- Sign up for **Free Account** (Beginner plan)

### 2. Upload Your App
- Use Web interface to upload files
- Or use Git: `git clone https://github.com/Josmarwen72/ai_10012300017`

### 3. Install Dependencies
```bash
pip install -r requirements-railway.txt
python scripts/build_index.py
```

### 4. Configure Web App
- Go to "Web" tab
- Add new "Web App"
- Choose "Flask"
- Point to your `app.py`

### 5. Benefits
- **Free tier** available
- **Reliable** platform
- **No payment** required for basic use
- **Python-focused** hosting

### 6. Limitations
- 2GB storage limit
- Some CPU/memory restrictions
- Your app may be slow with ML dependencies

### 7. URL Format
Your app will be at: `yourusername.pythonanywhere.com`
