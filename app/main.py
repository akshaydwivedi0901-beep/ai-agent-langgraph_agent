import os
from fastapi import FastAPI, UploadFile, File
from app.vectorstore import build_vectorstore, load_vectorstore
from app.llm import get_llm

app = FastAPI()

UPLOAD_DIR = "data/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    pdf_path = f"{UPLOAD_DIR}/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    chunks = build_vectorstore(pdf_path)
    return {"chunks": chunks}

@app.post("/chat")
def chat(message: str):
    db = load_vectorstore()
    llm = get_llm()

    docs = db.similarity_search(message, k=3)
    context = "\n".join(d.page_content for d in docs)

    response = llm.invoke(
        f"Answer using context:\n{context}\n\nQuestion: {message}"
    )

    return {"answer": response.content}
