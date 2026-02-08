# Render.com Deployment Guide

## Deploy to Render (Free)

1. Go to [render.com](https://render.com) and sign up
2. Click "New +" → "Web Service"
3. Connect your GitHub repository: `heeralal1806/document-qa-agent`
4. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python3 main.py`
5. Add Environment Variables:
   - `OPENAI_API_KEY` = your_openai_api_key
   - `GEMINI_API_KEY` = your_gemini_api_key (optional)
6. Click "Create Web Service"

## After Deployment

Your website will be available at: `https://document-qa-agent.onrender.com`

## Local Development

```bash
cd document-qa-agent
python3 main.py
# Open http://localhost:8000
```

## Repository

**GitHub:** https://github.com/heeralal1806/document-qa-agent

**Live Demo:** (Deploy to Render using steps above)

