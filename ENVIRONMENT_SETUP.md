# Environment Variables Setup

This project uses environment variables to securely manage sensitive configuration data like API keys.

## 🚀 Quick Setup

1. **Install dependencies** (includes python-dotenv):
   ```bash
   pip install -r requirements.txt
   ```

2. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env`** with your actual API keys:
   ```bash
   # Get your Groq API key from: https://console.groq.com/keys
   GROQ_API_KEY=your_groq_api_key_here
   
   # Optional: Get your OpenAI API key from: https://platform.openai.com/api-keys
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## 🔐 Security Benefits

- ✅ **No hardcoded secrets** in your code
- ✅ **Git-safe** - `.env` is excluded from version control
- ✅ **Easy deployment** - different configs for different environments
- ✅ **Team collaboration** - share `.env.example`, keep `.env` private

## 📁 Files Overview

- **`.env`** - Your private environment variables (DO NOT commit to Git)
- **`.env.example`** - Template showing required variables (SAFE to commit)
- **`backend/config.py`** - Loads environment variables automatically
- **`.gitignore`** - Excludes `.env` files from version control

## 🔧 Configuration Options

### LLM Providers
- `GROQ_API_KEY` - Groq API key (recommended for speed)
- `OPENAI_API_KEY` - OpenAI API key (optional)
- `ACITY_LLM_PROVIDER` - Choose: `groq`, `openai`, or `local`

### Retrieval Settings
- `ACITY_TOP_K` - Number of chunks to retrieve (default: 8)
- `ACITY_HYBRID_ALPHA` - Weight on dense similarity (default: 0.65)
- `ACITY_MAX_CONTEXT_CHARS` - Context limit for LLM (default: 6000)

## 🚦 Testing

Run the environment test to verify setup:
```bash
python test_env.py
```

Expected output:
```
✅ Environment Variables Test Results:
   LLM Provider: groq
   Groq API Key: ✅ Set
   OpenAI API Key: ❌ Not set
✅ Config loaded successfully from environment variables!
```

## 🔄 GitHub Push Safe

Your `.env` file is now excluded from Git, so you can safely push to GitHub without exposing your API keys!
