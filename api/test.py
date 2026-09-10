from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Vercel!", "status": "working"}

@app.get("/api/test")
def test():
    return JSONResponse({"test": "success", "platform": "vercel"})
