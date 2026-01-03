from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, shutil

from app.vectorstore import build_vectorstore, load_vectorstore
from app.llm import get_llm

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    os.makedirs("data/pdfs", exist_ok=True)
    path = f"data/pdfs/{file.filename}"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = build_vectorstore(path)
    return {"chunks_indexed": chunks}

@app.post("/chat")
def chat(req: ChatRequest):
    db = load_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(req.message)
    context = "\n\n".join(d.page_content for d in docs)

    llm = get_llm()
    response = llm.invoke(f"Context:\n{context}\n\nQuestion:\n{req.message}")

    return {"reply": response.content}

@app.get("/health")
def health():
    return {"status": "ok"}
