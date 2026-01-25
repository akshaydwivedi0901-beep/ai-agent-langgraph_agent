import os
from dotenv import load_dotenv
load_dotenv()
import time
import uuid
import logging
from typing import Generator

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import RedirectResponse, StreamingResponse

from app.vectorstore import build_vectorstore, load_vectorstore
from app.llm import get_llm
from app.memory import get_history, append_message
from app.cache import RedisCache
from app.safety import check_input_safety, check_output_safety

# =========================================================
# OBSERVABILITY CONFIG
# =========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("genai")

# =========================================================
# APP INIT
# =========================================================
app = FastAPI(title="Agentic RAG API")

BASE_DIR = os.getenv("APP_BASE_DIR", os.getcwd())
UPLOAD_DIR = os.path.join(BASE_DIR, "data/pdfs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

VECTORSTORE = None

# =========================================================
# REDIS CACHES (SHARED ACROSS PODS)
# =========================================================
RETRIEVAL_CACHE = RedisCache(prefix="retrieval", ttl_seconds=300)
PROMPT_CACHE = RedisCache(prefix="prompt", ttl_seconds=300)

# =========================================================
# MIDDLEWARE — REQUEST OBSERVABILITY
# =========================================================
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency_ms = int((time.time() - start_time) * 1000)
        logger.info({
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "latency_ms": latency_ms
        })


# =========================================================
# BASIC ROUTES
# =========================================================
@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================
# PDF UPLOAD (SYNC FOR NOW – ASYNC PIPELINE ALREADY DESIGNED)
# =========================================================
@app.post("/upload-pdf")
def upload_pdf(file: UploadFile = File(...)):
    global VECTORSTORE
    start_time = time.time()

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(pdf_path, "wb") as f:
            for chunk in iter(lambda: file.file.read(1024 * 1024), b""):
                f.write(chunk)

        chunks = build_vectorstore(pdf_path)
        VECTORSTORE = load_vectorstore()

        logger.info({
            "event": "pdf_upload",
            "filename": file.filename,
            "chunks": chunks,
            "latency_ms": int((time.time() - start_time) * 1000)
        })

        return {"status": "success", "chunks": chunks}

    except Exception as e:
        logger.error({"event": "pdf_upload_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="PDF processing failed")


# =========================================================
# LLM-AS-JUDGE (ANSWER QUALITY CHECK)
# =========================================================
def judge_answer(llm, question: str, context: str, answer: str) -> bool:
    judge_prompt = f"""
You are a strict answer-quality validator.

Question:
{question}

Context:
{context}

Answer:
{answer}

Decide if the answer:
- Is grounded in the context
- Does NOT hallucinate
- Directly answers the question

Reply with ONLY one word:
PASS or FAIL
"""
    result = llm.invoke(judge_prompt).content.strip().upper()
    return result == "PASS"


# =========================================================
# NORMAL CHAT (NON-STREAMING)
# =========================================================
@app.post("/chat")
def chat(message: str, session_id: str):
    global VECTORSTORE
    llm = get_llm()
    start_time = time.time()

    # ---------- INPUT SAFETY ----------
    safe, _ = check_input_safety(message)
    if not safe:
        raise HTTPException(status_code=400, detail="Unsafe input detected")

    try:
        if VECTORSTORE is None:
            VECTORSTORE = load_vectorstore()

        history = get_history(session_id)

        # ---------- RETRIEVAL CACHE ----------
        retrieval_key = f"{message}"
        docs = RETRIEVAL_CACHE.get(retrieval_key)

        if not docs:
            docs = VECTORSTORE.similarity_search(message, k=3)
            RETRIEVAL_CACHE.set(retrieval_key, docs)

        if not docs:
            raise HTTPException(status_code=404, detail="No relevant context found")

        context = "\n".join(d.page_content for d in docs)
        history_block = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history
        )

        prompt = f"{history_block}\n\nContext:\n{context}\n\nUser: {message}"

        # ---------- PROMPT CACHE ----------
        prompt_key = f"{session_id}:{hash(prompt)}"
        cached = PROMPT_CACHE.get(prompt_key)
        if cached:
            return {"answer": cached}

        # ---------- LLM CALL ----------
        response = llm.invoke(prompt)
        answer = response.content

        # ---------- OUTPUT SAFETY ----------
        safe, _ = check_output_safety(answer)
        if not safe:
            raise HTTPException(status_code=500, detail="Response blocked by safety")

        # ---------- QUALITY CHECK ----------
        if not judge_answer(llm, message, context, answer):
            logger.warning({"event": "judge_failed", "session_id": session_id})
            answer = "I’m not confident enough to answer that based on the document."

        # ---------- MEMORY ----------
        append_message(session_id, "user", message)
        append_message(session_id, "assistant", answer)

        PROMPT_CACHE.set(prompt_key, answer)

        logger.info({
            "event": "chat_complete",
            "latency_ms": int((time.time() - start_time) * 1000)
        })

        return {"answer": answer}

    except Exception as e:
        logger.error({"event": "chat_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="Chat failed")


# =========================================================
# STREAMING CHAT (SSE + REDIS MEMORY)
# =========================================================
@app.post("/chat/stream")
def chat_stream(message: str, session_id: str):
    global VECTORSTORE
    llm = get_llm(streaming=True)

    safe, _ = check_input_safety(message)
    if not safe:
        return StreamingResponse(
            iter(["data: Unsafe input detected\n\n"]),
            media_type="text/event-stream"
        )

    if VECTORSTORE is None:
        VECTORSTORE = load_vectorstore()

    history = get_history(session_id)

    def event_generator() -> Generator[str, None, None]:
        full_response = ""

        try:
            docs = VECTORSTORE.similarity_search(message, k=3)
            if not docs:
                yield "data: No relevant context found\n\n"
                return

            context = "\n".join(d.page_content for d in docs)
            history_block = "\n".join(
                f"{m['role'].capitalize()}: {m['content']}" for m in history
            )

            prompt = f"{history_block}\n\nContext:\n{context}\n\nUser: {message}\n\n"

            for chunk in llm.stream(prompt):
                if chunk.content:
                    full_response += chunk.content
                    yield f"data: {chunk.content}\n\n"

            # ---------- OUTPUT SAFETY ----------
            safe, _ = check_output_safety(full_response)
            if not safe:
                yield "data: Response blocked by safety policy\n\n"
                return

            # ---------- QUALITY CHECK ----------
            judge_llm = get_llm()
            if not judge_answer(judge_llm, message, context, full_response):
                yield "data: I’m not confident enough to answer that based on the document.\n\n"
                return

            append_message(session_id, "user", message)
            append_message(session_id, "assistant", full_response)

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error({"event": "chat_stream_error", "error": str(e)})
            yield "data: Error occurred\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
