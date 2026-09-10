import os
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
    """Build the RAG chain with uploaded documents"""
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
    """Main page with upload and chat interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TalentLens - RAG Resume Bot</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }
            h1 {
                color: #667eea;
                text-align: center;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                font-size: 1.1em;
                margin-bottom: 30px;
            }
            .status {
                padding: 15px;
                margin: 20px 0;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 1.1em;
            }
            .status.ready {
                background-color: #d4edda;
                color: #155724;
                border: 2px solid #28a745;
            }
            .status.not-ready {
                background-color: #f8d7da;
                color: #721c24;
                border: 2px solid #dc3545;
            }
            .section {
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            h3 {
                color: #667eea;
                margin-top: 0;
            }
            button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 1em;
                margin: 5px;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            button:active {
                transform: translateY(0);
            }
            input[type="file"] {
                margin: 10px 0;
                padding: 10px;
                border: 2px dashed #667eea;
                border-radius: 8px;
                width: 100%;
                cursor: pointer;
            }
            input[type="text"] {
                width: calc(100% - 120px);
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #667eea;
            }
            .chat-box {
                margin-top: 20px;
            }
            #chatHistory {
                max-height: 400px;
                overflow-y: auto;
                margin-top: 20px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                border: 1px solid #ddd;
            }
            .message {
                margin: 15px 0;
                padding: 12px;
                border-radius: 8px;
                animation: fadeIn 0.3s;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .user-message {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                text-align: right;
                margin-left: 20%;
            }
            .bot-message {
                background-color: #f1f8e9;
                border-left: 4px solid #4caf50;
                margin-right: 20%;
            }
            .loading {
                text-align: center;
                color: #667eea;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 TalentLens</h1>
            <p class="subtitle">RAG-Powered Resume Intelligence System</p>
            
            <div id="status" class="status not-ready">
                ⏳ Not Ready - Please upload resumes
            </div>
            
            <div class="section">
                <h3>📄 Upload Resumes</h3>
                <input type="file" id="fileInput" multiple accept=".pdf">
                <div style="margin-top: 10px;">
                    <button onclick="uploadFiles()">🚀 Upload & Process</button>
                    <button onclick="resetSystem()" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">🔄 Reset</button>
                </div>
            </div>
            
            <div class="section chat-box">
                <h3>💬 Ask Questions</h3>
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="questionInput" placeholder="e.g., Which candidate has the most Python experience?">
                    <button onclick="askQuestion()">Ask</button>
                </div>
                <div id="chatHistory"></div>
            </div>
        </div>
        
        <script>
            async function checkStatus() {
                try {
                    const response = await fetch('/api/status');
                    const data = await response.json();
                    const statusDiv = document.getElementById('status');
                    if (data.ready) {
                        statusDiv.className = 'status ready';
                        statusDiv.textContent = '✅ Ready - You can ask questions about the resumes';
                    } else {
                        statusDiv.className = 'status not-ready';
                        statusDiv.textContent = '⏳ Not Ready - Please upload resumes first';
                    }
                } catch (error) {
                    console.error('Status check failed:', error);
                }
            }
            
            async function uploadFiles() {
                const fileInput = document.getElementById('fileInput');
                const files = fileInput.files;
                
                if (files.length === 0) {
                    alert('⚠️ Please select PDF files first');
                    return;
                }
                
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                
                const statusDiv = document.getElementById('status');
                statusDiv.className = 'status';
                statusDiv.textContent = '⏳ Processing files... This may take a moment...';
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    if (response.ok) {
                        alert(`✅ Success! Processed ${data.files} file(s) into ${data.chunks} chunks`);
                        checkStatus();
                        fileInput.value = '';
                    } else {
                        alert('❌ Error: ' + data.error);
                        checkStatus();
                    }
                } catch (error) {
                    alert('❌ Upload failed: ' + error);
                    checkStatus();
                }
            }
            
            async function resetSystem() {
                if (!confirm('Are you sure you want to reset? This will clear all uploaded resumes and chat history.')) {
                    return;
                }
                
                try {
                    const response = await fetch('/api/reset', { method: 'POST' });
                    if (response.ok) {
                        alert('✅ System reset successfully');
                        document.getElementById('chatHistory').innerHTML = '';
                        document.getElementById('fileInput').value = '';
                        checkStatus();
                    }
                } catch (error) {
                    alert('❌ Reset failed: ' + error);
                }
            }
            
            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const question = input.value.trim();
                
                if (!question) {
                    alert('⚠️ Please enter a question');
                    return;
                }
                
                const chatHistory = document.getElementById('chatHistory');
                chatHistory.innerHTML += `<div class="message user-message"><strong>You:</strong> ${question}</div>`;
                chatHistory.innerHTML += `<div class="message loading">🤔 Thinking...</div>`;
                chatHistory.scrollTop = chatHistory.scrollHeight;
                
                try {
                    const response = await fetch('/api/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question })
                    });
                    const data = await response.json();
                    
                    // Remove loading message
                    const loadingMsg = chatHistory.querySelector('.loading');
                    if (loadingMsg) loadingMsg.remove();
                    
                    if (response.ok) {
                        chatHistory.innerHTML += `<div class="message bot-message"><strong>🤖 TalentLens:</strong> ${data.answer}</div>`;
                    } else {
                        chatHistory.innerHTML += `<div class="message bot-message"><strong>❌ Error:</strong> ${data.error}</div>`;
                    }
                    
                    input.value = '';
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                } catch (error) {
                    const loadingMsg = chatHistory.querySelector('.loading');
                    if (loadingMsg) loadingMsg.remove();
                    chatHistory.innerHTML += `<div class="message bot-message"><strong>❌ Error:</strong> ${error}</div>`;
                    chatHistory.scrollTop = chatHistory.scrollHeight;
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
    """Check if system is ready"""
    return {"ready": STATE["ready"]}


@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload and process resume PDFs"""
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})
    
    docs = []
    for f in files:
        # Save to temp directory
        temp_path = os.path.join(UPLOAD_DIR, f.filename)
        try:
            with open(temp_path, "wb") as out:
                content = await f.read()
                out.write(content)
            
            docs.extend(PyPDFLoader(temp_path).load())
            # Clean up temp file
            os.remove(temp_path)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to parse {f.filename}: {str(e)}"},
            )
    
    if not docs:
        return JSONResponse(status_code=400, content={"error": "No text extracted from PDFs."})
    
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
    """Reset the system state"""
    STATE["conversation"] = None
    STATE["ready"] = False
    return {"ok": True}


class Question(BaseModel):
    question: str


@app.post("/api/chat")
def chat(q: Question):
    """Ask questions about uploaded resumes"""
    if not STATE["conversation"]:
        return JSONResponse(
            status_code=400, content={"error": "Please upload and process resumes first."}
        )
    try:
        answer = STATE["conversation"]({"question": q.question})["answer"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Chat failed: {str(e)}"})
    return {"answer": answer}
