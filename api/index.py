import os
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

app = FastAPI(title="TalentLens")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Use temporary directory for uploads in serverless environment
UPLOAD_DIR = tempfile.gettempdir()

STATE = {"conversation": None, "ready": False}


def build_chain(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    # Get OpenAI API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return chain, len(chunks)


@app.get("/")
def index():
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TalentLens - RAG Resume Bot</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .status {
                padding: 10px;
                margin: 20px 0;
                border-radius: 5px;
                text-align: center;
            }
            .status.ready {
                background-color: #d4edda;
                color: #155724;
            }
            .status.not-ready {
                background-color: #f8d7da;
                color: #721c24;
            }
            button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                cursor: pointer;
                margin: 5px;
            }
            button:hover {
                background-color: #0056b3;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            .chat-box {
                margin-top: 20px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 5px;
            }
            .message {
                margin: 10px 0;
                padding: 10px;
                border-radius: 5px;
            }
            .user-message {
                background-color: #e3f2fd;
                text-align: right;
            }
            .bot-message {
                background-color: #f1f8e9;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 TalentLens</h1>
            <p style="text-align: center; color: #666;">RAG-Powered Resume Intelligence</p>
            
            <div id="status" class="status not-ready">
                Status: Not Ready - Please upload resumes
            </div>
            
            <div>
                <h3>Upload Resumes</h3>
                <input type="file" id="fileInput" multiple accept=".pdf">
                <button onclick="uploadFiles()">Upload & Process</button>
                <button onclick="resetSystem()">Reset</button>
            </div>
            
            <div class="chat-box">
                <h3>Ask Questions</h3>
                <input type="text" id="questionInput" placeholder="Ask about the resumes..." style="width: 70%; padding: 10px;">
                <button onclick="askQuestion()">Ask</button>
                <div id="chatHistory" style="margin-top: 20px;"></div>
            </div>
        </div>
        
        <script>
            async function checkStatus() {
                const response = await fetch('/api/status');
                const data = await response.json();
                const statusDiv = document.getElementById('status');
                if (data.ready) {
                    statusDiv.className = 'status ready';
                    statusDiv.textContent = 'Status: Ready - You can ask questions';
                } else {
                    statusDiv.className = 'status not-ready';
                    statusDiv.textContent = 'Status: Not Ready - Please upload resumes';
                }
            }
            
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;
                
                if (files.length === 0) {
                    alert('Please select files first');
                    return;
                }
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        alert(`Success! Processed ${data.files} files into ${data.chunks} chunks`);
                        checkStatus();
                    } else {
                        alert('Error: ' + data.error);
                    }
                } catch (error) {
                    alert('Upload failed: ' + error);
                }
            }
            
            async function resetSystem() {
                const response = await fetch('/api/reset', { method: 'POST' });
                if (response.ok) {
                    alert('System reset successfully');
                    document.getElementById('chatHistory').innerHTML = '';
                    checkStatus();
                }
            }
            
            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const question = input.value.trim();
                
                if (!question) {
                    alert('Please enter a question');
                    return;
                }
                
                const chatHistory = document.getElementById('chatHistory');
                chatHistory.innerHTML += `<div class="message user-message"><strong>You:</strong> ${question}</div>`;
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        chatHistory.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> ${data.answer}</div>`;
                    } else {
                        chatHistory.innerHTML += `<div class="message bot-message"><strong>Error:</strong> ${data.error}</div>`;
                    }
                    
                    input.value = '';
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                } catch (error) {
                    chatHistory.innerHTML += `<div class="message bot-message"><strong>Error:</strong> ${error}</div>`;
                }
            }
            
            // Check status on load
            checkStatus();
            
            // Allow Enter key to submit question
            document.getElementById('questionInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    askQuestion();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/api/status")
def status():
    return {"ready": STATE["ready"]}


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})
    
    docs = []
    for f in files:
        # Save to temp directory
        temp_path = os.path.join(UPLOAD_DIR, f.filename)
        with open(temp_path, "wb") as out:
            content = await f.read()
            out.write(content)
        
        try:
            docs.extend(PyPDFLoader(temp_path).load())
            # Clean up temp file
            os.remove(temp_path)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to parse {f.filename}: {str(e)}"},
            )
    
    if not docs:
        return JSONResponse(status_code=400, content={"error": "No text extracted."})
    
    try:
        chain, n_chunks = build_chain(docs)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}"},
        )
    
    STATE["conversation"] = chain
    STATE["ready"] = True
    return {"ok": True, "files": len(files), "chunks": n_chunks}


@app.post("/api/reset")
def reset():
    STATE["conversation"] = None
    STATE["ready"] = False
    return {"ok": True}


class Question(BaseModel):
    question: str


@app.post("/api/chat")
def chat(q: Question):
    if not STATE["conversation"]:
        return JSONResponse(
            status_code=400, content={"error": "Upload and process resumes first."}
        )
    try:
        answer = STATE["conversation"]({"question": q.question})["answer"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat failed: {str(e)}"})
    return {"answer": answer}


# Vercel expects the ASGI app to be exported
# The handler will be called by Vercel's Python runtime
