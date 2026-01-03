from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

VECTOR_DIR = "data/index"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def build_vectorstore(pdf_path: str) -> int:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)
    os.makedirs(VECTOR_DIR, exist_ok=True)
    db.save_local(VECTOR_DIR)

    return len(chunks)

def load_vectorstore():
    return FAISS.load_local(VECTOR_DIR, embeddings)
