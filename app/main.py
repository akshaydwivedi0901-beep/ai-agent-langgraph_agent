import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
from app.vectorstore import build_vectorstore, load_vectorstore
from app.llm import get_llm

app = FastAPI(title="Agentic RAG API")

BASE_DIR = os.getenv("APP_BASE_DIR", os.getcwd())
UPLOAD_DIR = os.path.join(BASE_DIR, "data/pdfs")
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

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # ✅ Stream write (safe for large files)
        with open(pdf_path, "wb") as f:
            for chunk in iter(lambda: file.file.read(1024 * 1024), b""):
                f.write(chunk)

        chunks = build_vectorstore(pdf_path)
        VECTORSTORE = load_vectorstore()

        return {
            "status": "success",
            "chunks": chunks,
            "message": "PDF processed successfully"
        }

    except ValueError as e:
        # Bad PDF / no text / empty chunks
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        print("❌ Upload error:", e)
        raise HTTPException(status_code=500, detail="Failed to process PDF")


@app.post("/chat")
def chat(message: str):
    global VECTORSTORE
    llm = get_llm()

    try:
        if VECTORSTORE is None:
            VECTORSTORE = load_vectorstore()

        docs = VECTORSTORE.similarity_search(message, k=3)

        if not docs:
            raise HTTPException(
                status_code=404,
                detail="No relevant context found. Upload a PDF first."
            )

        context = "\n".join(d.page_content for d in docs)

        response = llm.invoke(
            f"Answer using the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {message}"
        )

        return {"answer": response.content}

    except RuntimeError as e:
        # Vectorstore missing
        raise HTTPException(status_code=409, detail=str(e))

    except Exception as e:
        print("❌ Chat error:", e)
        raise HTTPException(status_code=500, detail="Chat failed")
