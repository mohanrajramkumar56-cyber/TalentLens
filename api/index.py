import os
import tempfile
from typing import List

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    
    # Use environment variable - let langchain-openai handle it
    os.environ["OPENAI_API_KEY"] = openai_api_key
    
    # Initialize without passing api_key parameter
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return chain, len(chunks)


@app.get("/")
def index():
    """Serve the original TalentLens UI"""
    try:
        html_path = os.path.join(os.path.dirname(__file__), "..", "index.html")
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=500)


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
