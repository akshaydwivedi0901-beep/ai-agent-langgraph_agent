import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
from app.vectorstore import build_vectorstore, load_vectorstore
from app.llm import get_llm

app = FastAPI(title="Agentic RAG API")

UPLOAD_DIR = "/app/data/pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

VECTORSTORE = None


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global VECTORSTORE

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(pdf_path, "wb") as f:
        f.write(file.file.read())

    chunks = build_vectorstore(pdf_path)
    VECTORSTORE = load_vectorstore()

    return {"chunks": chunks}


@app.post("/chat")
def chat(message: str):
    global VECTORSTORE
    llm = get_llm()

    if VECTORSTORE is None:
        VECTORSTORE = load_vectorstore()

    docs = VECTORSTORE.similarity_search(message, k=3)
    context = "\n".join(d.page_content for d in docs)

    response = llm.invoke(
        f"Answer using context:\n{context}\n\nQuestion: {message}"
    )

    return {"answer": response.content}
