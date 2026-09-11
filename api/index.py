import os
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from pypdf import PdfReader
from openai import OpenAI

app = FastAPI(title="TalentLens")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"documents": "", "ready": False}

def extract_text_from_pdf(file_path):
    """Extract text from PDF"""
    text = ""
    try:
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

@app.get("/")
def index():
    """Serve the UI"""
    try:
        ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading UI: {str(e)}</h1>")

@app.get("/api/status")
def status():
    """Check if system is ready"""
    return {"ready": STATE["ready"]}

@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload and process resume PDFs"""
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "OPENAI_API_KEY not set"})
    
    UPLOAD_DIR = tempfile.gettempdir()
    all_text = ""
    file_count = 0
    
    for f in files:
        temp_path = os.path.join(UPLOAD_DIR, f.filename)
        try:
            with open(temp_path, "wb") as out:
                content = await f.read()
                out.write(content)
            
            text = extract_text_from_pdf(temp_path)
            if text:
                all_text += f"\n\n--- {f.filename} ---\n{text}"
                file_count += 1
            
            os.remove(temp_path)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to process {f.filename}: {str(e)}"},
            )
    
    if not all_text:
        return JSONResponse(status_code=400, content={"error": "No text extracted."})
    
    STATE["documents"] = all_text
    STATE["ready"] = True
    
    return {"ok": True, "files": file_count, "chars": len(all_text)}

@app.post("/api/reset")
def reset():
    """Reset the system"""
    STATE["documents"] = ""
    STATE["ready"] = False
    return {"ok": True}

class Question(BaseModel):
    question: str

@app.post("/api/chat")
def chat(q: Question):
    """Ask a question about resumes using OpenAI"""
    if not STATE["ready"]:
        return JSONResponse(
            status_code=400, content={"error": "Please upload resumes first."}
        )
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "OPENAI_API_KEY not set"})
    
    try:
        client = OpenAI(api_key=api_key)
        
        prompt = f"""You are a helpful assistant analyzing resumes. 

Here are the resumes:
{STATE["documents"]}

User question: {q.question}

Please answer the question based on the resume content."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful resume analyzer assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
        )
        
        answer = response.choices[0].message.content
        return {"answer": answer}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat failed: {str(e)}"})
