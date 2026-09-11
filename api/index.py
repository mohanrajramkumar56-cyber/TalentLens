import os
import sys
import traceback

# Test imports first
try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    print("✓ FastAPI imports OK", file=sys.stderr)
except Exception as e:
    print(f"✗ FastAPI import failed: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from pydantic import BaseModel
    print("✓ Pydantic imports OK", file=sys.stderr)
except Exception as e:
    print(f"✗ Pydantic import failed: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    print("✓ LangChain OpenAI imports OK", file=sys.stderr)
except Exception as e:
    print(f"✗ LangChain OpenAI import failed: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("✓ LangChain text splitter imports OK", file=sys.stderr)
except Exception as e:
    print(f"✗ LangChain text splitter import failed: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
    print("✓ LangChain community imports OK", file=sys.stderr)
except Exception as e:
    print(f"✗ LangChain community import failed: {e}", file=sys.stderr)
    traceback.print_exc()

try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    print("✓ LangChain chains imports OK", file=sys.stderr)
except Exception as e:
    print(f"✗ LangChain chains import failed: {e}", file=sys.stderr)
    traceback.print_exc()

# Now actually import everything
import tempfile
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="TalentLens")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"conversation": None, "ready": False}

@app.get("/")
def index():
    """Root endpoint - serve the UI"""
    try:
        ui_path = os.path.join(os.path.dirname(__file__), "ui.html")
        with open(ui_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head><title>TalentLens</title></head>
<body>
    <h1>TalentLens</h1>
    <p>Error loading UI: {str(e)}</p>
    <p><a href="/api/test">Test API</a></p>
</body>
</html>
        """)

@app.get("/api/test")
def test():
    """Simple test endpoint"""
    return {"status": "ok", "message": "API is working", "python_version": sys.version}

@app.get("/api/status")
def status():
    """Check if system is ready"""
    return {"ready": STATE["ready"]}

@app.get("/api/test-imports")
def test_imports():
    """Test if all imports work"""
    results = {}
    
    try:
        from langchain_openai import OpenAIEmbeddings
        results["langchain_openai"] = "✓ OK"
    except Exception as e:
        results["langchain_openai"] = f"✗ {str(e)}"
    
    try:
        from langchain_community.vectorstores import FAISS
        results["faiss"] = "✓ OK"
    except Exception as e:
        results["faiss"] = f"✗ {str(e)}"
    
    try:
        from langchain.chains import ConversationalRetrievalChain
        results["langchain_chains"] = "✓ OK"
    except Exception as e:
        results["langchain_chains"] = f"✗ {str(e)}"
    
    api_key_set = os.getenv("OPENAI_API_KEY") is not None
    results["openai_api_key_set"] = "✓ Yes" if api_key_set else "✗ No"
    
    return results

@app.post("/api/reset")
def reset():
    """Reset the system state"""
    STATE["conversation"] = None
    STATE["ready"] = False
    return {"ok": True}

class Question(BaseModel):
    question: str

# Only import heavy libraries when needed
@app.post("/api/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload and process resume PDFs"""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import FAISS
        from langchain.chains import ConversationalRetrievalChain
        from langchain.memory import ConversationBufferMemory
        
        if not files:
            return JSONResponse(status_code=400, content={"error": "No files uploaded."})
        
        UPLOAD_DIR = tempfile.gettempdir()
        docs = []
        
        for f in files:
            temp_path = os.path.join(UPLOAD_DIR, f.filename)
            try:
                with open(temp_path, "wb") as out:
                    content = await f.read()
                    out.write(content)
                
                docs.extend(PyPDFLoader(temp_path).load())
                os.remove(temp_path)
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to parse {f.filename}: {str(e)}"},
                )
        
        if not docs:
            return JSONResponse(status_code=400, content={"error": "No text extracted from PDFs."})
        
        # Build chain
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            return JSONResponse(status_code=500, content={"error": "OPENAI_API_KEY not configured"})
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_store.as_retriever(), memory=memory
        )
        
        STATE["conversation"] = chain
        STATE["ready"] = True
        return {"ok": True, "files": len(files), "chunks": len(chunks)}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed: {str(e)}", "traceback": traceback.format_exc()},
        )

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
