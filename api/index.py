import os
import sys
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

# Simple test app to diagnose issues
app = FastAPI(title="TalentLens Debug")

@app.get("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TalentLens - Diagnostic Page</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
            .success { color: green; font-weight: bold; }
            .info { background: #f0f0f0; padding: 15px; border-radius: 5px; margin: 10px 0; }
            h1 { color: #333; }
        </style>
    </head>
    <body>
        <h1>🎯 TalentLens Diagnostic</h1>
        <p class="success">✅ FastAPI is working on Vercel!</p>
        
        <div class="info">
            <h3>System Info:</h3>
            <p>Python Version: """ + sys.version + """</p>
            <p>Check /api/env to see environment variables</p>
            <p>Check /api/test-import to test heavy imports</p>
        </div>
        
        <div class="info">
            <h3>Next Steps:</h3>
            <ol>
                <li>Verify OpenAI API key is set: <a href="/api/env">/api/env</a></li>
                <li>Test imports: <a href="/api/test-import">/api/test-import</a></li>
            </ol>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api/env")
def check_env():
    """Check if environment variables are set"""
    api_key = os.getenv("OPENAI_API_KEY")
    return {
        "openai_key_set": api_key is not None,
        "openai_key_length": len(api_key) if api_key else 0,
        "python_version": sys.version,
    }

@app.get("/api/test-import")
def test_imports():
    """Test if heavy imports work"""
    results = {}
    
    try:
        import fastapi
        results["fastapi"] = "✅ OK"
    except Exception as e:
        results["fastapi"] = f"❌ {str(e)}"
    
    try:
        import langchain
        results["langchain"] = "✅ OK"
    except Exception as e:
        results["langchain"] = f"❌ {str(e)}"
    
    try:
        import langchain_openai
        results["langchain_openai"] = "✅ OK"
    except Exception as e:
        results["langchain_openai"] = f"❌ {str(e)}"
    
    try:
        import langchain_text_splitters
        results["langchain_text_splitters"] = "✅ OK"
    except Exception as e:
        results["langchain_text_splitters"] = f"❌ {str(e)}"
    
    try:
        from langchain_community.vectorstores import FAISS
        results["faiss"] = "✅ OK"
    except Exception as e:
        results["faiss"] = f"❌ {str(e)}"
    
    try:
        from langchain_community.document_loaders import PyPDFLoader
        results["pypdf"] = "✅ OK"
    except Exception as e:
        results["pypdf"] = f"❌ {str(e)}"
    
    return results

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "traceback": traceback.format_exc()
        }
    )
