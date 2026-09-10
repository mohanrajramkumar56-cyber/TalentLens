import os
import shutil

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_resumes")

app = FastAPI(title="TalentLens")

STATE = {"conversation": None, "ready": False}


def build_chain(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_documents(chunks, embeddings)
    llm = Ollama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return chain, len(chunks)


@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "index.html"))


@app.get("/api/status")
def status():
    return {"ready": STATE["ready"]}


@app.post("/api/upload")
def upload(files: list[UploadFile] = File(...)):
    if not files:
        return JSONResponse(status_code=400, content={"error": "No files uploaded."})
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    docs = []
    for f in files:
        path = os.path.join(UPLOAD_DIR, f.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        try:
            docs.extend(PyPDFLoader(path).load())
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to parse {f.filename}: {e}"},
            )
    if not docs:
        return JSONResponse(status_code=400, content={"error": "No text extracted."})
    try:
        chain, n_chunks = build_chain(docs)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Processing failed (is Ollama running?): {e}"},
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
        return JSONResponse(status_code=500, content={"error": f"Chat failed: {e}"})
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)