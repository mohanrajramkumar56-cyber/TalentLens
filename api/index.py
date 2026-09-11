import os
import tempfile
import json
import re
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

STATE = {"documents": "", "ready": False, "skills": []}

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

def extract_skills(text):
    """Extract skills from resume text"""
    skills_keywords = [
        "python", "java", "javascript", "react", "angular", "nodejs", "express",
        "sql", "mongodb", "postgresql", "mysql", "aws", "docker", "kubernetes",
        "git", "linux", "windows", "html", "css", "typescript", "nodejs",
        "fastapi", "django", "flask", "spring", "microservices", "rest api",
        "graphql", "machine learning", "deep learning", "data science",
        "agile", "scrum", "project management", "leadership", "communication"
    ]
    
    text_lower = text.lower()
    found_skills = []
    
    for skill in skills_keywords:
        if skill in text_lower:
            found_skills.append(skill)
    
    return list(set(found_skills))

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
    STATE["skills"] = extract_skills(all_text)
    STATE["ready"] = True
    
    return {"ok": True, "files": file_count, "chars": len(all_text), "skills": STATE["skills"]}

@app.post("/api/reset")
def reset():
    """Reset the system"""
    STATE["documents"] = ""
    STATE["skills"] = []
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
        client = OpenAI(api_key=api_key, timeout=60.0)
        
        prompt = f"""You are a helpful assistant analyzing resumes. 

Here are the resumes:
{STATE["documents"][:5000]}

User question: {q.question}

Please answer the question based on the resume content. Be concise."""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful resume analyzer. Answer concisely."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
        )
        
        answer = response.choices[0].message.content
        return {"answer": answer}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat failed: {str(e)}"})

@app.post("/api/interview")
def generate_interview():
    """Generate interview questions INSTANTLY - no API calls"""
    if not STATE["ready"]:
        return JSONResponse(
            status_code=400, content={"error": "Please upload resumes first."}
        )
    
    skills = STATE["skills"]
    
    # TECHNICAL QUESTIONS based on detected skills
    technical_questions = [
        "Can you explain a challenging project you've worked on and how you solved the technical problems?",
        "What design patterns are you most familiar with and where have you applied them?",
        "Describe your experience with the technologies listed in your resume. Which one are you most proficient in?",
        "How do you approach debugging and troubleshooting code issues?",
        "Tell us about a time when you had to optimize code or improve performance. What did you do?",
        "How do you handle version control and collaborate with other developers?",
        "Describe your experience with databases. How do you design efficient database schemas?",
    ]
    
    # Add skill-specific questions
    if any(s in skills for s in ["python", "java", "javascript"]):
        technical_questions.extend([
            "Can you walk us through a complex algorithm you've implemented?",
            "How do you write and maintain unit tests for your code?",
            "What's your experience with API development and RESTful services?",
        ])
    
    if any(s in skills for s in ["react", "angular", "nodejs"]):
        technical_questions.extend([
            "How do you manage state in your frontend applications?",
            "What's your approach to handling async operations and promises?",
        ])
    
    if any(s in skills for s in ["docker", "kubernetes", "aws"]):
        technical_questions.extend([
            "How do you approach containerization and deployment?",
            "What's your experience with cloud platforms and microservices?",
        ])
    
    # GENERAL/BEHAVIORAL QUESTIONS
    general_questions = [
        "Tell me about yourself and your professional background.",
        "What are your greatest strengths as a developer/professional?",
        "What areas would you like to improve or develop further?",
        "Describe a time when you had to work with a difficult team member. How did you handle it?",
        "Tell us about a project where you took the lead. What was the outcome?",
        "How do you stay updated with new technologies and industry trends?",
        "Describe a situation where you had to learn something new quickly. How did you approach it?",
        "Tell us about a failure or mistake you made and what you learned from it.",
        "How do you prioritize your work when you have multiple tasks?",
        "What are your career goals for the next 3-5 years?",
        "Why are you interested in this position/company?",
        "How do you communicate technical concepts to non-technical stakeholders?",
        "Tell us about your experience working in Agile environments.",
        "How do you handle feedback and criticism?",
        "Describe your ideal work environment and team culture.",
    ]
    
    return {
        "technical_questions": technical_questions[:10],
        "general_questions": general_questions,
        "detected_skills": skills,
        "total_questions": 25
    }
