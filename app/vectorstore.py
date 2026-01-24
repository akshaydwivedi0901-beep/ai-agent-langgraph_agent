import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

VECTOR_DIR = "/app/data/vectorstore"

embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

def build_vectorstore(pdf_path: str) -> int:
    """
    Load a PDF, split it into chunks, generate embeddings, and save FAISS index.
    Returns: number of chunks processed.
    """
    os.makedirs(VECTOR_DIR, exist_ok=True)

    # 1️⃣ Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    if not docs:
        raise ValueError("PDF contains no readable pages")

    # 2️⃣ Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    # 🔴 CRITICAL GUARD
    if not chunks:
        raise ValueError("No text could be extracted from the PDF")

    # 3️⃣ Build vectorstore
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(VECTOR_DIR)

    print(f"✅ Vectorstore built with {len(chunks)} chunks")
    return len(chunks)


def load_vectorstore():
    """
    Load an existing FAISS vectorstore from disk.
    """
    index_path = os.path.join(VECTOR_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise RuntimeError("Vectorstore not found. Upload a PDF first.")

    print(f"✅ Loading vectorstore from {VECTOR_DIR}")
    return FAISS.load_local(
        VECTOR_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
